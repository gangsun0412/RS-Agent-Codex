"""
意图识别模块
负责判断用户请求类型，抽取关键参数，判断参数完整性，输出结构化 JSON
"""
import json
import os
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional
from openai import OpenAI


class Intent(str, Enum):
    REMOTE_SENSING_PROCESSING = "remote_sensing_processing"
    STANDARD_QA = "standard_qa"
    PAPER_SEARCH = "paper_search"
    CHAT = "chat"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """意图识别结果（对标设计文档 6.1 节）"""
    intent: str
    confidence: float
    task_name: str = ""
    slots: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    need_user_clarification: bool = False
    clarification_question: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ============ 意图分类提示词 ============
INTENT_CLASSIFICATION_PROMPT = """你是一个遥感领域智能助手的意图分类器。请分析用户输入，判断意图类型并抽取参数。

## 意图类型说明：

1. **remote_sensing_processing**（遥感数据处理）
   - 用户要执行遥感数据处理任务，如：预处理、地形校正、模型训练、NDVI估算、时序重建
   - 关键词：GLASS、NDVI、Landsat、地形校正、随机森林、时序重建、预处理、裁剪等
   - 示例：
     - "帮我处理2020年的GLASS NDVI数据"
     - "对Landsat影像做地形校正"
     - "训练随机森林模型估算NDVI"

2. **standard_qa**（标准规范查询）
   - 用户查询遥感、测绘、无人机等相关国家标准或技术规范
   - 关键词：标准、规范、要求、规定、质量检查、精度、无人机航测等
   - 示例：
     - "无人机航测的航向重叠度要求是多少？"
     - "数字测绘成果质量检查有哪些内容？"

3. **paper_search**（论文检索）
   - 用户要搜索或检索学术论文、研究文献
   - 关键词：论文、文献、检索、搜索、最新研究、综述等
   - 示例：
     - "搜索NDVI时序重建相关论文"
     - "最近有哪些随机森林遥感应用的文献？"

4. **chat**（闲聊/解释性问答）
   - 一般性对话、概念解释、系统使用帮助等
   - 示例：
     - "什么是NDVI？"
     - "这个系统能做什么？"
     - "你好"

5. **unknown**（无法判断）
   - 模糊不清或无法归类为以上任一类型

## 参数槽位（仅 remote_sensing_processing 需要抽取）：

- input_hdf_folder: HDF数据文件夹
- output_folder: 输出文件夹
- roi_shp_path: ROI矢量文件路径
- year_start: 起始年份
- year_end: 结束年份
- input_sr_folder: Landsat SR影像文件夹
- slope_file: 坡度文件
- aspect_file: 坡向文件
- landsat_folder: Landsat数据文件夹
- glass_folder: GLASS数据文件夹
- model_save_path: 模型保存路径
- model_path: 模型文件路径
- parameter_folder: 参数矩阵文件夹

## 输出格式（严格JSON，不要有额外文本）：

{
  "intent": "意图类型",
  "confidence": 0.0-1.0的置信度,
  "task_name": "任务简称（如ndvi_reconstruction）",
  "slots": {"已抽取的参数名": "参数值"},
  "missing_slots": ["缺失的必填参数名"],
  "need_user_clarification": true/false,
  "clarification_question": "如果缺参，向用户追问的具体问题"
}
"""


class IntentClassifier:
    """意图分类器：使用 LLM 判断用户意图并抽取参数"""

    def __init__(self, client: OpenAI, model: str = "deepseek-chat"):
        """
        Args:
            client: OpenAI 兼容客户端
            model: LLM 模型名
        """
        self.client = client
        self.model = model

    def classify(self, user_input: str) -> IntentResult:
        """
        对用户输入进行意图分类和参数抽取

        Args:
            user_input: 用户自然语言输入

        Returns:
            IntentResult 结构化结果
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1,  # 低温度，提高分类一致性
                max_tokens=500
            )

            raw_output = response.choices[0].message.content.strip()

            # 清理可能的 markdown 代码块标记
            if raw_output.startswith("```"):
                raw_output = raw_output.split("\n", 1)[-1]
                if raw_output.endswith("```"):
                    raw_output = raw_output[:-3]
                raw_output = raw_output.strip()

            data = json.loads(raw_output)

            return IntentResult(
                intent=data.get("intent", Intent.UNKNOWN),
                confidence=float(data.get("confidence", 0.0)),
                task_name=data.get("task_name", ""),
                slots=data.get("slots", {}),
                missing_slots=data.get("missing_slots", []),
                need_user_clarification=data.get("need_user_clarification", False),
                clarification_question=data.get("clarification_question", "")
            )

        except (json.JSONDecodeError, KeyError) as e:
            # LLM 返回格式非法时，回退到 unknown
            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                need_user_clarification=True,
                clarification_question="抱歉，我没有理解您的需求。请更具体地描述您想做什么？"
            )
        except Exception as e:
            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                need_user_clarification=True,
                clarification_question=f"系统异常（{str(e)}），请重试或换一种方式描述。"
            )

    def route(self, result: IntentResult) -> str:
        """
        根据意图识别结果决定路由策略

        Returns:
            路由目标标识: "tool", "rag", "paper", "chat", "unknown"
        """
        if result.intent == Intent.REMOTE_SENSING_PROCESSING:
            return "tool"
        elif result.intent == Intent.STANDARD_QA:
            return "rag"
        elif result.intent == Intent.PAPER_SEARCH:
            return "paper"
        elif result.intent == Intent.CHAT:
            return "chat"
        else:
            return "unknown"


# ============ 测试 ============
if __name__ == "__main__":
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    if not API_KEY:
        print("请设置 DEEPSEEK_API_KEY 环境变量")
        exit(1)

    client = OpenAI(
        api_key=API_KEY,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    classifier = IntentClassifier(client)

    test_inputs = [
        "帮我处理2020到2022年的GLASS NDVI数据，HDF在D:\\data\\hdf",
        "无人机航测的重叠度有什么要求？",
        "搜索NDVI时序重建的最新论文",
        "什么是随机森林算法？",
        "asdfghjkl",
    ]

    for inp in test_inputs:
        result = classifier.classify(inp)
        route = classifier.route(result)
        print(f"\n输入: {inp}")
        print(f"意图: {result.intent} (置信度: {result.confidence})")
        print(f"路由: {route}")
        if result.need_user_clarification:
            print(f"追问: {result.clarification_question}")
