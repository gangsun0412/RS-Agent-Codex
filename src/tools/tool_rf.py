"""
随机森林训练与应用工具
功能：基于 Landsat SR 和 GLASS NDVI 训练 RF 回归模型，并应用模型估算晴空 NDVI
"""
import os
import json
import matlab.engine


def rf_train(
    landsat_folder: str,
    glass_folder: str,
    model_save_path: str,
    matlab_script_dir: str
) -> str:
    """
    随机森林训练Tool：训练RF模型并保存为.mat文件

    Args:
        landsat_folder: PLC校正后Landsat SR文件夹
        glass_folder: 预处理后GLASS NDVI文件夹
        model_save_path: 模型保存完整路径，含文件名(.mat)
        matlab_script_dir: Matlab脚本所在目录

    Returns:
        JSON字符串，包含status、model_path、message
    """
    try:
        if not os.path.isdir(landsat_folder):
            raise FileNotFoundError(f"Landsat文件夹不存在: {landsat_folder}")
        if not os.path.isdir(glass_folder):
            raise FileNotFoundError(f"GLASS文件夹不存在: {glass_folder}")

        print("[RF训练] 启动Matlab引擎...")
        eng = matlab.engine.start_matlab()
        eng.addpath(matlab_script_dir, nargout=0)

        print("[RF训练] 开始训练...")
        eng.rf_train(
            landsat_folder,
            glass_folder,
            model_save_path,
            nargout=0
        )
        eng.quit()

        result = {
            "status": "success",
            "model_path": model_save_path,
            "message": f"RF模型训练完成，已保存至{model_save_path}"
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        result = {"status": "error", "message": str(e)}
        return json.dumps(result, ensure_ascii=False)


def rf_apply(
    landsat_folder: str,
    model_path: str,
    output_folder: str,
    matlab_script_dir: str
) -> str:
    """
    随机森林应用Tool：用训练好的模型估算晴空NDVI

    Args:
        landsat_folder: PLC校正后Landsat SR文件夹
        model_path: 训练好的RF模型路径(.mat)
        output_folder: 晴空NDVI输出文件夹
        matlab_script_dir: Matlab脚本所在目录

    Returns:
        JSON字符串，包含status、output_folder、message
    """
    try:
        if not os.path.isdir(landsat_folder):
            raise FileNotFoundError(f"Landsat文件夹不存在: {landsat_folder}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        os.makedirs(output_folder, exist_ok=True)

        print("[RF应用] 启动Matlab引擎...")
        eng = matlab.engine.start_matlab()
        eng.addpath(matlab_script_dir, nargout=0)

        print("[RF应用] 开始估算晴空NDVI...")
        eng.rf_apply(
            landsat_folder,
            model_path,
            output_folder,
            nargout=0
        )
        eng.quit()

        result = {
            "status": "success",
            "output_folder": output_folder,
            "message": f"晴空NDVI估算完成，结果保存在{output_folder}"
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        result = {"status": "error", "message": str(e)}
        return json.dumps(result, ensure_ascii=False)


# ====== 单独测试用 ======
if __name__ == "__main__":
    MATLAB_SCRIPT_DIR = r"E:\MatlabCode\TALESF_Codex"

    # 测试应用（训练完再跑）
    result = rf_apply(
        landsat_folder=r"G:\C_RF训练_Test\TC_LC08_SR_utm47_clip_nan",
        model_path=r"D:\Codex_code\RF_model.mat",
        output_folder=r"D:\data\rf_output",
        matlab_script_dir=MATLAB_SCRIPT_DIR
    )
    print(result)
