"""
PLC 地形校正工具
功能：对 Landsat SR 影像进行地形辐射校正，通过 matlab.engine 调用 Matlab 脚本
"""
import os
import json


def run_plc_correction(
        input_sr_folder: str,
        slope_file: str,
        aspect_file: str,
        output_folder: str,
        matlab_script_dir: str
) -> str:
    """
    PLC地形校正工具：通过matlab.engine调用Matlab脚本完成地形校正

    Args:
        input_sr_folder: Landsat SR影像输入文件夹
        slope_file: 坡度文件路径(.tif)
        aspect_file: 坡向文件路径(.tif)
        output_folder: PLC校正结果输出文件夹
        matlab_script_dir: Matlab脚本所在目录

    Returns:
        JSON字符串，包含status、output_folder、message
    """
    try:
        # 延迟导入重型依赖
        import matlab.engine

        if not os.path.isdir(input_sr_folder):
            raise FileNotFoundError(f"SR影像文件夹不存在: {input_sr_folder}")
        if not os.path.isfile(slope_file):
            raise FileNotFoundError(f"坡度文件不存在: {slope_file}")
        if not os.path.isfile(aspect_file):
            raise FileNotFoundError(f"坡向文件不存在: {aspect_file}")

        os.makedirs(output_folder, exist_ok=True)

        print("[PLC] 启动Matlab引擎...")
        eng = matlab.engine.start_matlab()
        eng.addpath(matlab_script_dir, nargout=0)

        print("[PLC] 开始地形校正...")
        eng.plc_correction(
            input_sr_folder,
            slope_file,
            aspect_file,
            output_folder,
            nargout=0
        )

        eng.quit()

        result = {
            "status": "success",
            "output_folder": output_folder,
            "message": f"PLC地形校正完成，结果保存在{output_folder}"
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
    MATLAB_SCRIPT_DIR = r"E:\MatlabCode\TALESF_Codex"
    result = run_plc_correction(
        input_sr_folder=r"G:\B_地形校正_0623_TEST\01SR_utm47",
        slope_file=r"G:\B_地形校正_0623_TEST\02slope_utm47\slope_utm47.tif",
        aspect_file=r"G:\B_地形校正_0623_TEST\03aspect_utm47\aspect_utm47.tif",
        output_folder=r"G:\B_地形校正_0623_TEST\04PLC_SR_utm47",
        matlab_script_dir=MATLAB_SCRIPT_DIR,
    )
    print(result)
