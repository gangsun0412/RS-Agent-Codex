"""
GLASS 数据下载工具
从 https://glass.hku.hk/archive 下载 GLASS 产品数据

URL 拼接模式:
    https://glass.hku.hk/archive/{product}/{source}/{resolution}/{year}/{doy}/{filename}
    filename = {prefix}.A{year}{doy}.{tile}.{product_date}.hdf

注意: prefix 和 product_date 不会预设，需要用户提供或在交互模式中追问。
      不同 DOY 的产品日期可能不同，请以服务器实际文件名为准。

使用方式:
    python -m src.tools.tool_glass_download    # 交互式下载
    python tests/test_glass_download.py         # 逻辑测试
"""
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# GLASS 服务器
BASE_URL = "https://glass.hku.hk/archive"

# 已知 MODIS 全球分块列表（中国及周边常用）
KNOWN_TILES = [
    "h23v04", "h23v05", "h24v04", "h24v05", "h25v03", "h25v04",
    "h25v05", "h25v06", "h26v03", "h26v04", "h26v05", "h26v06",
    "h27v04", "h27v05", "h27v06", "h28v05", "h28v06", "h29v05",
    "h29v06",
]


# ====================================================================
# URL 构建 —— 所有参数必须显式提供，无预设
# ====================================================================

def build_glass_urls(
    product: str,
    source: str,
    resolution: str,
    year_start: int,
    year_end: int,
    tiles: list,
    prefix: str,
    product_date: str,
    doy_step: int = 8,
    doy_start: int = 1,
    doy_end: int = 361,
    base_url: str = BASE_URL,
) -> list:
    """
    构建 GLASS 数据下载链接列表。所有参数均为必填，不做预设。

    Args:
        product:     产品路径名，如 NDVI, LAI, GPP
        source:      数据来源，如 MODIS, AVHRR
        resolution:  空间分辨率，如 250M, 500M, 1KM
        year_start:  起始年份
        year_end:    结束年份
        tiles:       目标分块列表，如 ['h26v05']
        prefix:      文件名前缀，如 GLASS13D01.V10
        product_date: 文件名末尾产品日期，如 2023062
        doy_step:    DOY 步长（默认 8 天）
        doy_start:   DOY 起始值（默认 1）
        doy_end:     DOY 结束值（默认 361）
        base_url:    GLASS 服务器基础地址

    Returns:
        [{"url": ..., "filename": ..., "year": ..., "doy": ..., "tile": ...}, ...]
    """
    doy_list = list(range(doy_start, doy_end + 1, doy_step))
    url_list = []

    for year in range(year_start, year_end + 1):
        year_str = str(year)
        for doy in doy_list:
            doy_str = f"{doy:03d}"
            for tile in tiles:
                filename = f"{prefix}.A{year_str}{doy_str}.{tile}.{product_date}.hdf"
                url = f"{base_url}/{product}/{source}/{resolution}/{year_str}/{doy_str}/{filename}"
                url_list.append({
                    "url": url,
                    "filename": filename,
                    "year": year,
                    "doy": doy,
                    "tile": tile,
                })
    return url_list


# ====================================================================
# URL 探测 —— 下载前验证链接是否有效
# ====================================================================

def probe_url(url: str, timeout: int = 15) -> bool:
    """用 HEAD 请求探测 URL 是否可访问"""
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        return resp.status_code == 200
    except Exception:
        return False


def probe_example(
    product: str,
    source: str,
    resolution: str,
    tile: str,
    prefix: str,
    product_date: str,
    year: int,
    doy: int = 1,
) -> dict:
    """
    探测一个示例文件是否存在。

    Returns:
        {"ok": bool, "url": str, "status": int, "dir_url": str}
    """
    doy_str = f"{doy:03d}"
    filename = f"{prefix}.A{year}{doy_str}.{tile}.{product_date}.hdf"
    url = f"{BASE_URL}/{product}/{source}/{resolution}/{year}/{doy_str}/{filename}"
    dir_url = f"{BASE_URL}/{product}/{source}/{resolution}/{year}/{doy_str}/"

    print(f"\n[Probe] 探测示例链接...")
    print(f"  {url}")
    try:
        resp = requests.head(url, timeout=15, allow_redirects=True)
        ok = resp.status_code == 200
        if ok:
            print(f"  [OK] 状态码 200")
        else:
            print(f"  [FAIL] 状态码 {resp.status_code}")
        return {"ok": ok, "url": url, "status": resp.status_code, "dir_url": dir_url}
    except Exception as e:
        print(f"  [FAIL] 连接失败: {e}")
        return {"ok": False, "url": url, "status": -1, "dir_url": dir_url}


# ====================================================================
# 文件校验
# ====================================================================

def _is_valid_hdf(filepath: str, dataset_name: str = "NDVI_250M") -> bool:
    """检查 HDF 文件是否包含目标数据集"""
    try:
        from pyhdf.SD import SD
        sd = SD(filepath)
        sd.select(dataset_name)
        sd.end()
        return True
    except Exception:
        pass
    try:
        from osgeo import gdal
        ds = gdal.Open(filepath)
        if ds is not None:
            subdatasets = ds.GetSubDatasets()
            ds = None
            return len(subdatasets) > 0
        return False
    except Exception:
        return False


# ====================================================================
# 下载引擎
# ====================================================================

def _download_one(url: str, save_path: str) -> tuple:
    """下载单个文件，返回 (path, ok, error)"""
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path, True, None
    except Exception as e:
        return save_path, False, str(e)


def download_files(
    url_list: list,
    output_dir: str,
    dataset_name: str = "NDVI_250M",
    max_workers: int = 8,
    skip_existing: bool = True,
) -> dict:
    """
    并行下载 + 断点续传 + 损坏重试。
    """
    os.makedirs(output_dir, exist_ok=True)

    to_download = []
    skipped = 0
    for item in url_list:
        save_path = os.path.join(output_dir, str(item["year"]), item["filename"])
        item["save_path"] = save_path
        if skip_existing and os.path.exists(save_path):
            if _is_valid_hdf(save_path, dataset_name):
                skipped += 1
                continue
        to_download.append(item)

    total = len(url_list)
    if not to_download:
        print(f"[OK] 全部 {total} 个文件已存在且有效，无需下载。")
        return {"total": total, "downloaded": 0, "skipped": skipped, "failed": 0, "failed_list": []}

    print(f"[DL] 待下载: {len(to_download)} 个文件 (已跳过 {skipped} 个)")

    downloaded = 0
    failed_list = []
    workers = min(max_workers, len(to_download))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, item["url"], item["save_path"]): item
            for item in to_download
        }
        for i, future in enumerate(as_completed(futures), 1):
            save_path, ok, err = future.result()
            if ok:
                downloaded += 1
                if i % 20 == 0 or i == len(to_download):
                    print(f"  [{i}/{len(to_download)}] [OK] {os.path.basename(save_path)}")
            else:
                failed_list.append({"path": save_path, "error": err})
                print(f"  [{i}/{len(to_download)}] [FAIL] {os.path.basename(save_path)} - {err[:80]}")

    # 重试
    if failed_list:
        print(f"\n[Retry] 重试 {len(failed_list)} 个失败文件...")
        still_failed = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for item in failed_list:
                for orig in to_download:
                    if orig["save_path"] == item["path"]:
                        futures[executor.submit(_download_one, orig["url"], item["path"])] = item
                        break
            for future in as_completed(futures):
                save_path, ok, err = future.result()
                if ok:
                    downloaded += 1
                else:
                    still_failed.append({"path": save_path, "error": err})
        failed_list = still_failed

    # 二次校验
    if failed_list:
        print(f"\n[Check] 二次校验...")
        for item in failed_list[:]:
            if os.path.exists(item["path"]) and _is_valid_hdf(item["path"], dataset_name):
                failed_list.remove(item)
                downloaded += 1

    result = {
        "total": total,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": len(failed_list),
        "failed_list": failed_list,
    }

    print(f"\n{'=' * 50}")
    print(f"[Stats] 总计 {total} | 新下载 {downloaded} | 跳过 {skipped} | 失败 {len(failed_list)}")
    if failed_list:
        print(f"[WARN] 以下文件下载失败，请检查:")
        for f in failed_list:
            print(f"  - {os.path.basename(f['path'])}")
            print(f"    错误: {f['error'][:120]}")
    return result


# ====================================================================
# 交互式入口 —— 每个不确定的参数都会追问
# ====================================================================

def glass_download(
    product: Optional[str] = None,
    source: Optional[str] = None,
    resolution: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    tiles: Optional[list] = None,
    prefix: Optional[str] = None,
    product_date: Optional[str] = None,
    doy_step: int = 8,
    output_dir: Optional[str] = None,
    auto_probe: bool = True,
    dataset_name: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    GLASS 数据下载主入口。所有可选参数未提供时会在控制台追问。

    特别说明: prefix 和 product_date 不会自动预设——
    因为同一天目录下不同 DOY 的产品日期可能不同，必须以服务器实际文件名为准。

    流程: 追问参数 → 探测示例 URL → 确认 → 下载
    """
    print("\n" + "=" * 50)
    print("  GLASS 数据下载工具")
    print("  官网: https://glass.hku.hk/archive")
    print("=" * 50)

    # ---------- 产品 ----------
    if product is None:
        print("\n[INFO] 可选产品 (更多见官网):")
        print("  NDVI / LAI / GPP / FPAR / ALBEDO / LST / ...")
        product = input("> 产品名 (如 NDVI): ").strip()
        if not product:
            print("[FAIL] 未输入产品，退出。")
            return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0, "failed_list": []}

    # ---------- 来源 ----------
    if source is None:
        source = input("> 数据来源 (MODIS/AVHRR, 回车=MODIS): ").strip() or "MODIS"

    # ---------- 分辨率 ----------
    if resolution is None:
        print("[INFO] 常见分辨率: 250M, 500M, 1KM, 0.05D")
        resolution = input("> 空间分辨率 (回车=250M): ").strip() or "250M"

    # ---------- 年份 ----------
    if year_start is None:
        try:
            year_start = int(input("> 起始年份 (如 2013): ").strip() or "2013")
        except ValueError:
            year_start = 2013
    if year_end is None:
        try:
            year_end = int(input("> 结束年份 (如 2020): ").strip() or "2020")
        except ValueError:
            year_end = 2020

    # ---------- 分块 ----------
    if tiles is None:
        print(f"\n[INFO] MODIS 全球分块 (中国及周边常用):")
        print(f"  {', '.join(KNOWN_TILES)}")
        print(f"  更多分块请访问官网查看。")
        user_input = input("> 分块 (多个逗号分隔, 如 h26v05,h27v05): ").strip()
        if not user_input:
            print("[FAIL] 未输入分块，退出。")
            return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0, "failed_list": []}
        tiles = [t.strip() for t in user_input.split(",") if t.strip()]

    # ---------- prefix ----------
    if prefix is None:
        dir_url = f"{BASE_URL}/{product}/{source}/{resolution}/{year_start}/001/"
        print(f"\n[INFO] 文件名格式: {{prefix}}.A{{year}}{{doy}}.{{tile}}.{{date}}.hdf")
        print(f"[INFO] 请浏览目录查看实际 prefix (产品版本号):")
        print(f"  {dir_url}")
        prefix = input("> prefix (如 GLASS13D01.V10): ").strip()
        if not prefix:
            print("[FAIL] 未输入 prefix，退出。")
            return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0, "failed_list": []}

    # ---------- product_date ----------
    if product_date is None:
        dir_url = f"{BASE_URL}/{product}/{source}/{resolution}/{year_start}/001/"
        print(f"\n[INFO] product_date = 文件名末尾的发布日期数字")
        print(f"[INFO] [WARN] 不同 DOY 的日期可能不同! (001→2023062, 014→2023063)")
        print(f"[INFO] 请浏览目录查看实际日期:")
        print(f"  {dir_url}")
        product_date = input("> product_date (如 2023062): ").strip()
        if not product_date:
            print("[FAIL] 未输入 product_date，退出。")
            return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0, "failed_list": []}

    # ---------- 输出目录 ----------
    if output_dir is None:
        default_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "raw", "GLASS"
        ))
        output_dir = input(f"> 输出目录 (回车={default_dir}): ").strip() or default_dir

    # ---------- 数据集名 (HDF校验用) ----------
    if dataset_name is None:
        if product.upper() == "NDVI":
            dataset_name = "NDVI_250M"
        elif product.upper() == "LAI":
            dataset_name = "LAI"
        else:
            dataset_name = input("> HDF数据集名 (用于校验, 如 NDVI_250M): ").strip() or product.upper()

    # ---------- 确认 ----------
    print(f"\n{'=' * 50}")
    print(f"[INFO] 参数确认:")
    print(f"  产品: {product}  |  来源: {source}  |  分辨率: {resolution}")
    print(f"  年份: {year_start}-{year_end}  |  分块: {tiles}")
    print(f"  prefix: {prefix}  |  product_date: {product_date}")
    print(f"  DOY步长: {doy_step}天  |  数据集: {dataset_name}")
    print(f"  输出: {output_dir}")
    print(f"{'=' * 50}")

    # ---------- 探测 ----------
    if auto_probe:
        result = probe_example(
            product=product,
            source=source,
            resolution=resolution,
            tile=tiles[0],
            prefix=prefix,
            product_date=product_date,
            year=year_start,
            doy=1,
        )
        if not result["ok"]:
            print(f"\n[FAIL] 示例 URL 探测失败!")
            print(f"  尝试的 URL: {result['url']}")
            print(f"  HTTP 状态码: {result['status']}")
            print(f"\n[INFO] 请按以下步骤排查:")
            print(f"  1. 在浏览器中打开以下目录:")
            print(f"     {result['dir_url']}")
            print(f"  2. 查看实际文件名格式，核对 prefix 和 product_date")
            print(f"  3. 重新运行，输入正确的参数")
            print(f"\n[INFO] 如果整个路径结构变了，请提供目录下的实际文件名示例，我来调整代码。")
            return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0, "failed_list": []}

    # ---------- 最终确认 ----------
    confirm = input(f"\n> 确认开始下载? (y/n, 回车=y): ").strip().lower()
    if confirm == "n":
        print("[INFO] 已取消。")
        return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0, "failed_list": []}

    # ---------- 构建 URL 并下载 ----------
    url_list = build_glass_urls(
        product=product,
        source=source,
        resolution=resolution,
        year_start=year_start,
        year_end=year_end,
        tiles=tiles,
        prefix=prefix,
        product_date=product_date,
        doy_step=doy_step,
        **kwargs,
    )
    print(f"\n[Link] 共生成 {len(url_list)} 个下载链接")
    print(f"  示例: {url_list[0]['url']}")

    result = download_files(url_list, output_dir, dataset_name=dataset_name)
    return result


# ====================================================================
if __name__ == "__main__":
    glass_download()
