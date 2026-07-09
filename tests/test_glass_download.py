"""
GLASS 下载工具 -- 测试脚本

用法:
    python tests/test_glass_download.py           # 纯逻辑测试 (不需联网)
    python tests/test_glass_download.py --probe   # 测试 URL 探测 (需联网)
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from src.tools.tool_glass_download import (
    build_glass_urls,
    probe_example,
    KNOWN_TILES,
    BASE_URL,
)


def test_url_construction():
    """测试 URL 构建逻辑 (不需联网)"""
    print("=" * 60)
    print("[Test] URL 构建")
    print("=" * 60)

    # 共用参数
    base_args = {
        "product": "NDVI",
        "source": "MODIS",
        "resolution": "250M",
        "year_start": 2013,
        "year_end": 2013,
        "tiles": ["h26v05"],
        "prefix": "GLASS13D01.V10",
        "product_date": "2023062",
    }

    # ---- 测试1: 基本格式 ----
    print("\n--- 测试1: 基本 URL 格式 ---")
    urls = build_glass_urls(**base_args)
    assert len(urls) == 46, f"预期 46 个 (365/8≈46), 实际 {len(urls)}"
    first = urls[0]
    print(f"  数量: {len(urls)}")
    print(f"  URL: {first['url']}")
    print(f"  文件: {first['filename']}")
    assert first["url"] == (
        "https://glass.hku.hk/archive/NDVI/MODIS/250M/2013/001/"
        "GLASS13D01.V10.A2013001.h26v05.2023062.hdf"
    )
    assert first["filename"] == "GLASS13D01.V10.A2013001.h26v05.2023062.hdf"
    print("  [OK]")

    # ---- 测试2: 多年多块 ----
    print("\n--- 测试2: 多年多块 ---")
    urls = build_glass_urls(**{**base_args, "year_end": 2014, "tiles": ["h26v05", "h27v05"]})
    expected = 46 * 2 * 2  # 46 DOYs × 2 tiles × 2 years
    assert len(urls) == expected, f"预期 {expected}, 实际 {len(urls)}"
    tile_set = set(u["tile"] for u in urls)
    assert tile_set == {"h26v05", "h27v05"}
    print(f"  数量: {len(urls)} = 46×2×2  [OK]")

    # ---- 测试3: 不同分辨率 ----
    print("\n--- 测试3: 不同分辨率 / 不同来源 ---")
    urls = build_glass_urls(**{**base_args, "resolution": "500M"})
    assert all("500M" in u["url"] for u in urls)
    print("  500M: [OK]")

    urls = build_glass_urls(**{**base_args, "source": "AVHRR"})
    assert all("AVHRR" in u["url"] for u in urls)
    print("  AVHRR: [OK]")

    # ---- 测试4: 自定义 prefix 和 product_date ----
    print("\n--- 测试4: 自定义 prefix/date ---")
    urls = build_glass_urls(**{**base_args, "prefix": "CUSTOM.V02", "product_date": "2025001"})
    assert urls[0]["filename"].startswith("CUSTOM.V02")
    assert "2025001" in urls[0]["filename"]
    print(f"  文件: {urls[0]['filename']}  [OK]")

    # ---- 测试5: 自定义 DOY 范围 ----
    print("\n--- 测试5: 自定义 DOY 范围和步长 ---")
    urls = build_glass_urls(**{**base_args, "doy_start": 1, "doy_end": 9, "doy_step": 8})
    assert len(urls) == 2  # DOY 1 和 9
    doy_set = {u["doy"] for u in urls}
    assert doy_set == {1, 9}
    print(f"  DOYs: {doy_set}  [OK]")

    # ---- 测试6: 缺失必填参数 ----
    print("\n--- 测试6: 缺 prefix/product_date 不抛异常 (由调用方处理) ---")
    try:
        build_glass_urls(product="NDVI", source="MODIS", resolution="250M",
                         year_start=2013, year_end=2013, tiles=["h26v05"],
                         prefix="", product_date="")
        print("  空字符串未报错，由交互层校验  [OK]")
    except Exception as e:
        print(f"  异常: {e}  (可接受)")


def test_known_tiles():
    """测试已知分块列表"""
    print("\n" + "=" * 60)
    print("[Test] 已知分块")
    print("=" * 60)
    import re
    pattern = re.compile(r"^h\d{2}v\d{2}$")
    for tile in KNOWN_TILES:
        assert pattern.match(tile), f"格式异常: {tile}"
    print(f"  {len(KNOWN_TILES)} 个分块，格式全部正确  [OK]")


def test_probe_url():
    """测试 URL 探测 (需联网)"""
    print("\n" + "=" * 60)
    print("[Test] URL 探测")
    print("=" * 60)
    result = probe_example(
        product="NDVI", source="MODIS", resolution="250M",
        tile="h26v05", prefix="GLASS13D01.V10", product_date="2023062",
        year=2020, doy=1,
    )
    if result["ok"]:
        print("  [OK] 服务器返回 200，URL 格式正确")
    else:
        print(f"  [WARN] HTTP {result['status']} - 需确认 URL 格式")
    print(f"  目录: {result['dir_url']}")


# ====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GLASS 下载工具测试")
    parser.add_argument("--probe", action="store_true", help="测试 URL 探测 (需联网)")
    args = parser.parse_args()

    test_url_construction()
    test_known_tiles()

    if args.probe:
        test_probe_url()

    print("\n" + "=" * 60)
    print("[OK] 全部逻辑测试通过!")
    print("=" * 60)
