"""
工具模块测试
测试参数校验、返回结构和异常处理
注意：部分测试需要在有 GDAL/MATLAB 的环境中运行
"""
import os
import sys
import json
import pytest

# 确保项目根目录在路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# 检测可用依赖
try:
    import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False

requires_gdal = pytest.mark.skipif(not HAS_GDAL, reason="需要 GDAL 库")
requires_matlab = pytest.mark.skipif(not HAS_MATLAB, reason="需要 MATLAB 引擎")


class TestToolPreprocess:
    """GLASS NDVI 预处理工具测试"""

    @requires_gdal
    def test_import(self):
        """测试模块可正常导入"""
        from src.tools.tool_preprocess import preprocess_glass_ndvi
        assert callable(preprocess_glass_ndvi)

    @requires_gdal
    def test_parse_glass_date_valid(self):
        """测试日期解析 - 正常文件名"""
        from src.tools.tool_preprocess import parse_glass_date
        result = parse_glass_date("GLASS13D01.V10.A2017233.h26v04.2023064.hdf")
        assert result == "20170821"

    @requires_gdal
    def test_parse_glass_date_invalid(self):
        """测试日期解析 - 无效文件名"""
        from src.tools.tool_preprocess import parse_glass_date
        with pytest.raises(ValueError):
            parse_glass_date("invalid_filename.hdf")

    @requires_gdal
    def test_preprocess_missing_input(self):
        """测试输入路径不存在时的错误处理"""
        from src.tools.tool_preprocess import preprocess_glass_ndvi
        result = preprocess_glass_ndvi(
            input_hdf_folder=r"Z:\nonexistent\path",
            output_folder=r"D:\tmp\test_output",
            roi_shp_path=r"D:\tmp\test.shp",
            year_start=2020,
            year_end=2020
        )
        data = json.loads(result)
        assert data["status"] == "error"

    def test_date_parsing_logic(self):
        """测试日期解析的核心逻辑（不导入 gdal）"""
        # 直接测试算法而非导入模块
        from datetime import datetime, timedelta

        def parse_glass_date(hdf_filename: str) -> str:
            parts = hdf_filename.split('.')
            for part in parts:
                if part.startswith('A') and len(part) == 8:
                    year = int(part[1:5])
                    day_of_year = int(part[5:8])
                    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
                    return date.strftime('%Y%m%d')
            raise ValueError(f"无法从文件名中解析日期: {hdf_filename}")

        assert parse_glass_date("GLASS13D01.V10.A2017233.h26v04.2023064.hdf") == "20170821"

        with pytest.raises(ValueError):
            parse_glass_date("bad_file.hdf")


class TestToolPLC:
    """PLC 地形校正工具测试"""

    @requires_matlab
    def test_import(self):
        """测试模块可正常导入"""
        from src.tools.tool_plc import run_plc_correction
        assert callable(run_plc_correction)

    @requires_matlab
    def test_missing_input_folder(self):
        """测试输入文件夹不存在时出错"""
        from src.tools.tool_plc import run_plc_correction
        result = run_plc_correction(
            input_sr_folder=r"Z:\nonexistent",
            slope_file=r"D:\tmp\slope.tif",
            aspect_file=r"D:\tmp\aspect.tif",
            output_folder=r"D:\tmp\plc_output",
            matlab_script_dir=r"D:\tmp"
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "不存在" in data["message"]


class TestToolRF:
    """随机森林工具测试"""

    @requires_matlab
    def test_import(self):
        """测试模块可正常导入"""
        from src.tools.tool_rf import rf_train, rf_apply
        assert callable(rf_train)
        assert callable(rf_apply)

    @requires_matlab
    def test_rf_train_missing_folder(self):
        """测试训练输入缺失时出错"""
        from src.tools.tool_rf import rf_train
        result = rf_train(
            landsat_folder=r"Z:\nonexistent",
            glass_folder=r"Z:\nonexistent2",
            model_save_path=r"D:\tmp\model.mat",
            matlab_script_dir=r"D:\tmp"
        )
        data = json.loads(result)
        assert data["status"] == "error"

    @requires_matlab
    def test_rf_apply_missing_model(self):
        """测试模型文件不存在时出错"""
        from src.tools.tool_rf import rf_apply
        result = rf_apply(
            landsat_folder=r"D:\tmp",
            model_path=r"Z:\nonexistent\model.mat",
            output_folder=r"D:\tmp\rf_output",
            matlab_script_dir=r"D:\tmp"
        )
        data = json.loads(result)
        assert data["status"] == "error"


class TestToolReconstruct:
    """NDVI 时序重建工具测试"""

    @requires_matlab
    def test_import(self):
        """测试模块可正常导入"""
        from src.tools.tool_reconstruct import reconstruct_ndvi
        assert callable(reconstruct_ndvi)

    @requires_matlab
    def test_missing_folder(self):
        """测试输入缺失时出错"""
        from src.tools.tool_reconstruct import reconstruct_ndvi
        result = reconstruct_ndvi(
            landsat_folder=r"Z:\nonexistent",
            glass_folder=r"Z:\nonexistent2",
            output_folder=r"D:\tmp\reconstruct_output",
            parameter_folder=r"D:\tmp\params",
            matlab_script_dir=r"D:\tmp"
        )
        data = json.loads(result)
        assert data["status"] == "error"


class TestToolBase:
    """工具基类测试（无需任何外部依赖）"""

    def test_tool_result_success(self):
        """测试成功结果工厂方法"""
        from src.tools.tool_base import ToolResult
        result = ToolResult.success_result(
            tool_name="test_tool",
            message="测试完成",
            outputs=["file1.tif", "file2.tif"],
            metadata={"count": 2}
        )
        assert result.success is True
        assert result.tool_name == "test_tool"
        assert len(result.outputs) == 2
        assert result.error is None
        assert result.metadata["count"] == 2

    def test_tool_result_error(self):
        """测试错误结果工厂方法"""
        from src.tools.tool_base import ToolResult
        result = ToolResult.error_result(
            tool_name="test_tool",
            message="执行失败",
            error="FileNotFoundError"
        )
        assert result.success is False
        assert result.error == "FileNotFoundError"
        assert len(result.outputs) == 0

    def test_tool_result_json(self):
        """测试 JSON 序列化"""
        from src.tools.tool_base import ToolResult
        result = ToolResult.success_result("test", "done")
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["success"] is True
        assert data["tool_name"] == "test"
