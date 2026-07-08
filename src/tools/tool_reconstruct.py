"""
NDVI 时序重建工具
功能：基于线性回归加权重建云覆盖缺失像元，生成连续8天时序NDVI产品
"""
import os
import json
import matlab.engine


def reconstruct_ndvi(
    landsat_folder: str,
    glass_folder: str,
    output_folder: str,
    parameter_folder: str,
    matlab_script_dir: str
) -> str:
    """
    NDVI时序重建Tool：基于线性回归加权重建缺失像元（16天→8天）

    Args:
        landsat_folder: RF估算后的Landsat NDVI文件夹
        glass_folder: 预处理后GLASS NDVI文件夹
        output_folder: 重建结果输出文件夹
        parameter_folder: 回归参数矩阵保存文件夹
        matlab_script_dir: Matlab脚本所在目录

    Returns:
        JSON字符串，包含status、output_folder、message
    """
    try:
        if not os.path.isdir(landsat_folder):
            raise FileNotFoundError(f"Landsat文件夹不存在: {landsat_folder}")
        if not os.path.isdir(glass_folder):
            raise FileNotFoundError(f"GLASS文件夹不存在: {glass_folder}")

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(parameter_folder, exist_ok=True)

        print("[重建] 启动Matlab引擎...")
        eng = matlab.engine.start_matlab()
        eng.addpath(matlab_script_dir, nargout=0)

        print("[重建] 开始NDVI时序重建...")
        eng.reconstruct_ndvi(
            landsat_folder,
            glass_folder,
            output_folder,
            parameter_folder,
            nargout=0
        )
        eng.quit()

        result = {
            "status": "success",
            "output_folder": output_folder,
            "message": f"NDVI时序重建完成，结果保存在{output_folder}"
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        result = {"status": "error", "message": str(e)}
        return json.dumps(result, ensure_ascii=False)


# ====== 单独测试用 ======
if __name__ == "__main__":
    MATLAB_SCRIPT_DIR = r"E:\MatlabCode\TALESF_Codex"

    result = reconstruct_ndvi(
        landsat_folder=r"D:\Codex_code\data\rf_output",
        glass_folder=r"G:\C_RF训练_Test\CD_GLASS_NDVI_utm47_clip_ymd",
        output_folder=r"D:\Codex_code\reconstruct_output",
        parameter_folder=r"D:\Codex_code",
        matlab_script_dir=MATLAB_SCRIPT_DIR
    )
    print(result)
