"""
意图识别模块测试
测试 IntentClassifier 的逻辑和路由决策（不依赖 LLM API）
"""
import os
import sys
import json
import pytest

# 确保项目根目录在路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from src.agent.intent_classifier import (
    Intent, IntentResult, IntentClassifier, INTENT_CLASSIFICATION_PROMPT
)


class TestIntentEnum:
    """意图枚举测试"""

    def test_all_intents_defined(self):
        """验证所有 5 种意图已定义"""
        assert Intent.REMOTE_SENSING_PROCESSING == "remote_sensing_processing"
        assert Intent.STANDARD_QA == "standard_qa"
        assert Intent.PAPER_SEARCH == "paper_search"
        assert Intent.CHAT == "chat"
        assert Intent.UNKNOWN == "unknown"

    def test_intent_is_string(self):
        """验证 Intent 可当字符串使用"""
        assert isinstance(Intent.CHAT, str)


class TestIntentResult:
    """意图识别结果数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        result = IntentResult(
            intent=Intent.CHAT,
            confidence=0.8
        )
        assert result.intent == "chat"
        assert result.confidence == 0.8
        assert result.task_name == ""
        assert result.slots == {}
        assert result.missing_slots == []
        assert result.need_user_clarification is False
        assert result.clarification_question == ""

    def test_with_slots(self):
        """测试带参数槽位的结果"""
        result = IntentResult(
            intent=Intent.REMOTE_SENSING_PROCESSING,
            confidence=0.9,
            task_name="ndvi_preprocess",
            slots={"year_start": 2020, "year_end": 2022},
            missing_slots=["input_hdf_folder", "output_folder", "roi_shp_path"],
            need_user_clarification=True,
            clarification_question="请提供HDF数据文件夹路径、输出文件夹和ROI文件。"
        )
        assert result.task_name == "ndvi_preprocess"
        assert result.slots["year_start"] == 2020
        assert len(result.missing_slots) == 3
        assert result.need_user_clarification is True

    def test_to_dict(self):
        """测试序列化为字典"""
        result = IntentResult(intent=Intent.CHAT, confidence=0.5)
        d = result.to_dict()
        assert d["intent"] == "chat"
        assert d["confidence"] == 0.5

    def test_to_json(self):
        """测试序列化为 JSON"""
        result = IntentResult(intent=Intent.CHAT, confidence=0.5)
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["intent"] == "chat"


class TestIntentClassifierRouting:
    """路由逻辑测试（不调用 LLM）"""

    def test_route_tool(self):
        """测试遥感处理 → tool 路由"""
        classifier = IntentClassifier(None)  # client=None, 不调用LLM
        result = IntentResult(
            intent=Intent.REMOTE_SENSING_PROCESSING,
            confidence=0.9
        )
        assert classifier.route(result) == "tool"

    def test_route_rag(self):
        """测试标准查询 → rag 路由"""
        classifier = IntentClassifier(None)
        result = IntentResult(intent=Intent.STANDARD_QA, confidence=0.8)
        assert classifier.route(result) == "rag"

    def test_route_paper(self):
        """测试论文检索 → paper 路由"""
        classifier = IntentClassifier(None)
        result = IntentResult(intent=Intent.PAPER_SEARCH, confidence=0.7)
        assert classifier.route(result) == "paper"

    def test_route_chat(self):
        """测试聊天 → chat 路由"""
        classifier = IntentClassifier(None)
        result = IntentResult(intent=Intent.CHAT, confidence=0.95)
        assert classifier.route(result) == "chat"

    def test_route_unknown(self):
        """测试未知意图路由"""
        classifier = IntentClassifier(None)
        result = IntentResult(intent=Intent.UNKNOWN, confidence=0.1)
        assert classifier.route(result) == "unknown"

    def test_route_unknown_low_confidence(self):
        """测试低置信度也应正确路由"""
        classifier = IntentClassifier(None)
        result = IntentResult(intent=Intent.UNKNOWN, confidence=0.0)
        assert classifier.route(result) == "unknown"


class TestIntentClassificationPrompt:
    """意图分类提示词测试"""

    def test_prompt_contains_all_intents(self):
        """验证提示词包含所有意图类型"""
        assert "remote_sensing_processing" in INTENT_CLASSIFICATION_PROMPT
        assert "standard_qa" in INTENT_CLASSIFICATION_PROMPT
        assert "paper_search" in INTENT_CLASSIFICATION_PROMPT
        assert "chat" in INTENT_CLASSIFICATION_PROMPT
        assert "unknown" in INTENT_CLASSIFICATION_PROMPT

    def test_prompt_has_output_format(self):
        """验证提示词包含输出格式说明"""
        assert "intent" in INTENT_CLASSIFICATION_PROMPT
        assert "confidence" in INTENT_CLASSIFICATION_PROMPT
        assert "missing_slots" in INTENT_CLASSIFICATION_PROMPT
        assert "clarification_question" in INTENT_CLASSIFICATION_PROMPT
