"""
工具基类模块
定义统一的工具返回结构和抽象基类，确保所有 Tool 接口一致
"""
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from abc import ABC, abstractmethod


@dataclass
class ToolResult:
    """统一的工具执行结果结构（对标设计文档 6.3 节）"""
    success: bool
    tool_name: str
    message: str
    outputs: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def success_result(tool_name: str, message: str, outputs: list = None,
                       metadata: dict = None) -> "ToolResult":
        """工厂方法：创建成功结果"""
        return ToolResult(
            success=True,
            tool_name=tool_name,
            message=message,
            outputs=outputs or [],
            metadata=metadata or {},
            error=None
        )

    @staticmethod
    def error_result(tool_name: str, message: str, error: str = None) -> "ToolResult":
        """工厂方法：创建失败结果"""
        return ToolResult(
            success=False,
            tool_name=tool_name,
            message=message,
            outputs=[],
            metadata={},
            error=error or message
        )


class BaseTool(ABC):
    """遥感处理工具抽象基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def get_schema(self) -> dict:
        """
        返回 OpenAI Function Calling 格式的 tool schema。
        子类必须实现此方法。

        Returns:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """
        pass

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """
        执行工具核心逻辑。子类必须实现此方法。

        Returns:
            ToolResult 统一结果对象
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> Optional[str]:
        """
        校验输入参数。返回错误描述字符串，校验通过返回 None。
        子类必须实现此方法。
        """
        pass

    def execute(self, **kwargs) -> str:
        """
        完整的工具执行流程：校验 → 执行 → 返回 JSON

        Returns:
            ToolResult JSON 字符串
        """
        start_time = time.time()

        # 1. 输入校验
        validation_error = self.validate_inputs(**kwargs)
        if validation_error:
            result = ToolResult.error_result(
                tool_name=self.name,
                message=validation_error,
                error=validation_error
            )
            return result.to_json()

        # 2. 执行
        try:
            result = self.run(**kwargs)
        except Exception as e:
            result = ToolResult.error_result(
                tool_name=self.name,
                message=f"工具执行异常: {str(e)}",
                error=str(e)
            )

        # 3. 附加耗时信息
        elapsed = round(time.time() - start_time, 2)
        result.metadata["elapsed_seconds"] = elapsed

        return result.to_json()

    def summarize_result(self, result: ToolResult) -> str:
        """生成用户友好的结果摘要（可被子类覆盖）"""
        if result.success:
            return f"✅ {result.message}"
        else:
            return f"❌ {result.message}"
