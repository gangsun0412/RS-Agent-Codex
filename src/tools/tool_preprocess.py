"""
GLASS NDVI 预处理工具
功能：HDF转TIFF → 重投影(EPSG:4326) → 重采样(240m) → 按ROI裁剪
"""
import os
import json
from datetime import datetime, timedelta
from osgeo import gdal

gdal.UseExceptions()


def parse_glass_date(hdf_filename: str) -> str:
    """
    从GLASS HDF文件名中提取日期
    例如：GLASS13D01.V10.A2017233.h26v04.2023064.hdf → 20170821
    """
    parts = hdf_filename.split('.')
    for part in parts:
        if part.startswith('A') and len(part) == 8:
            year = int(part[1:5])
            day_of_year = int(part[5:8])
            date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
            return date.strftime('%Y%m%d')
    raise ValueError(f"无法从文件名中解析日期: {hdf_filename}")


def preprocess_glass_ndvi(
        input_hdf_folder: str,
        output_folder: str,
        roi_shp_path: str,
        year_start: int,
        year_end: int
) -> str:
    """
    GLASS NDVI预处理工具：HDF转TIFF → 重投影(EPSG:4326) → 重采样(240m) → 裁剪

    Args:
        input_hdf_folder: HDF文件根目录，按年份组织子文件夹
        output_folder: 预处理结果输出根目录
        roi_shp_path: 裁剪用ROI矢量文件路径(.shp)
        year_start: 处理起始年份
        year_end: 处理结束年份

    Returns:
        JSON字符串，包含status、output_folder、message
    """
    try:
        # 定义输出子文件夹
        tiff_folder = os.path.join(output_folder, "1_tiff")
        project_folder = os.path.join(output_folder, "2_projected")
        resample_folder = os.path.join(output_folder, "3_resampled")
        clip_folder = os.path.join(output_folder, "4_clipped")

        for folder in [tiff_folder, project_folder, resample_folder, clip_folder]:
            os.makedirs(folder, exist_ok=True)

        nodata_value = 25500  # GLASS NDVI无效值

        for year in range(year_start, year_end + 1):
            year_hdf_path = os.path.join(input_hdf_folder, str(year))
            if not os.path.isdir(year_hdf_path):
                print(f"[跳过] 年份文件夹不存在: {year_hdf_path}")
                continue

            print(f"\n[处理年份] {year}")

            # ====== Step1: HDF → TIFF ======
            year_tiff_path = os.path.join(tiff_folder, str(year))
            os.makedirs(year_tiff_path, exist_ok=True)

            for hdf_file in os.listdir(year_hdf_path):
                if not hdf_file.lower().endswith('.hdf'):
                    continue

                hdf_full_path = os.path.join(year_hdf_path, hdf_file)
                date_str = parse_glass_date(hdf_file)
                tiff_full_path = os.path.join(year_tiff_path, f"{date_str}.tif")

                if os.path.exists(tiff_full_path):
                    print(f"  [跳过已存在] {tiff_full_path}")
                    continue

                ds = gdal.Open(hdf_full_path)
                if ds is None:
                    print(f"  [警告] 无法打开: {hdf_full_path}")
                    continue

                gdal.Translate(
                    tiff_full_path, ds,
                    format='GTiff',
                    noData=nodata_value
                )
                ds = None
                print(f"  [HDF→TIFF] {hdf_file}")

            # ====== Step2: 重投影 → EPSG:4326 ======
            year_project_path = os.path.join(project_folder, str(year))
            os.makedirs(year_project_path, exist_ok=True)

            for tif_file in os.listdir(year_tiff_path):
                if not tif_file.lower().endswith('.tif'):
                    continue

                src_path = os.path.join(year_tiff_path, tif_file)
                dst_path = os.path.join(year_project_path, tif_file)

                if os.path.exists(dst_path):
                    print(f"  [跳过已存在] {dst_path}")
                    continue

                gdal.Warp(
                    dst_path, src_path,
                    dstSRS='EPSG:4326',
                    resampleAlg=gdal.GRA_NearestNeighbour,
                    srcNodata=nodata_value,
                    dstNodata=nodata_value
                )
                print(f"  [重投影] {tif_file}")

            # ====== Step3: 重采样 → 240m ======
            year_resample_path = os.path.join(resample_folder, str(year))
            os.makedirs(year_resample_path, exist_ok=True)

            res_deg = 240 / 111320.0

            for tif_file in os.listdir(year_project_path):
                if not tif_file.lower().endswith('.tif'):
                    continue

                src_path = os.path.join(year_project_path, tif_file)
                dst_path = os.path.join(year_resample_path, tif_file)

                if os.path.exists(dst_path):
                    print(f"  [跳过已存在] {dst_path}")
                    continue

                gdal.Warp(
                    dst_path, src_path,
                    xRes=res_deg, yRes=res_deg,
                    resampleAlg=gdal.GRA_NearestNeighbour,
                    srcNodata=nodata_value,
                    dstNodata=nodata_value
                )
                print(f"  [重采样] {tif_file}")

            # ====== Step4: 裁剪 → ROI ======
            year_clip_path = os.path.join(clip_folder, str(year))
            os.makedirs(year_clip_path, exist_ok=True)

            for tif_file in os.listdir(year_resample_path):
                if not tif_file.lower().endswith('.tif'):
                    continue

                src_path = os.path.join(year_resample_path, tif_file)
                dst_path = os.path.join(year_clip_path, tif_file)

                if os.path.exists(dst_path):
                    print(f"  [跳过已存在] {dst_path}")
                    continue

                gdal.Warp(
                    dst_path, src_path,
                    cutlineDSName=roi_shp_path,
                    cropToCutline=True,
                    srcNodata=nodata_value,
                    dstNodata=nodata_value
                )
                print(f"  [裁剪] {tif_file}")

        result = {
            "status": "success",
            "output_folder": clip_folder,
            "message": f"{year_start}-{year_end}年GLASS NDVI预处理完成，结果保存在{clip_folder}"
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        result = {
            "status": "error",
            "message": str(e)
        }
        return json.dumps(result, ensure_ascii=False)


# ====== 单独测试用 ======
if __name__ == "__main__":
    result = preprocess_glass_ndvi(
        input_hdf_folder=r"C:\Users\30662\Desktop\hdf-test",
        output_folder=r"C:\Users\30662\Desktop\preprocess_test",
        roi_shp_path=r"G:\滑坡植被扰动\WC_juxingFW\Juxing_shp\test.shp",
        year_start=2017,
        year_end=2017
    )
    print(result)
