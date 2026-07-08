"""
Agent 核心框架
负责对话管理、意图识别、参数抽取、工具调用分发和结果解释
"""
import os
import json
import sys
from openai import OpenAI

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.tools.tool_preprocess import preprocess_glass_ndvi
from src.tools.tool_plc import run_plc_correction
from src.tools.tool_rf import rf_train, rf_apply
from src.tools.tool_reconstruct import reconstruct_ndvi
from src.agent.intent_classifier import IntentClassifier, Intent, IntentResult


# ============ 配置（从环境变量读取） ============
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# MATLAB脚本目录：优先环境变量，其次项目内 matlab_scripts/
MATLAB_SCRIPT_DIR = os.getenv(
    "MATLAB_SCRIPT_DIR",
    os.path.join(_PROJECT_ROOT, "matlab_scripts")
)


# ============ OpenAI 客户端 ============
def _create_client() -> OpenAI:
    """创建OpenAI兼容客户端"""
    if not API_KEY:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY，请设置环境变量或创建 .env 文件。\n"
            "参考 .env.example 文件进行配置。"
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============ 工具定义 ============
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "preprocess_glass_ndvi",
            "description": "GLASS NDVI预处理：HDF转TIFF、重投影到EPSG:4326、重采样到240m、按ROI裁剪",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_hdf_folder": {"type": "string", "description": "HDF文件根目录，按年份组织子文件夹"},
                    "output_folder": {"type": "string", "description": "预处理结果输出根目录"},
                    "roi_shp_path": {"type": "string", "description": "裁剪用ROI矢量文件路径(.shp)"},
                    "year_start": {"type": "integer", "description": "处理起始年份"},
                    "year_end": {"type": "integer", "description": "处理结束年份"}
                },
                "required": ["input_hdf_folder", "output_folder", "roi_shp_path", "year_start", "year_end"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_plc_correction",
            "description": "PLC地形校正：对Landsat SR影像进行地形辐射校正，消除地形引起的辐射误差",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_sr_folder": {"type": "string", "description": "Landsat SR影像输入文件夹"},
                    "slope_file": {"type": "string", "description": "坡度文件路径(.tif)"},
                    "aspect_file": {"type": "string", "description": "坡向文件路径(.tif)"},
                    "output_folder": {"type": "string", "description": "PLC校正结果输出文件夹"}
                },
                "required": ["input_sr_folder", "slope_file", "aspect_file", "output_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rf_train",
            "description": "随机森林模型训练：基于Landsat SR和GLASS NDVI训练RF回归模型，用于估算晴空NDVI",
            "parameters": {
                "type": "object",
                "properties": {
                    "landsat_folder": {"type": "string", "description": "PLC校正后Landsat SR文件夹"},
                    "glass_folder": {"type": "string", "description": "预处理后GLASS NDVI文件夹"},
                    "model_save_path": {"type": "string", "description": "模型保存完整路径，含文件名(.mat)"}
                },
                "required": ["landsat_folder", "glass_folder", "model_save_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rf_apply",
            "description": "随机森林模型应用：用训练好的RF模型估算Landsat晴空NDVI",
            "parameters": {
                "type": "object",
                "properties": {
                    "landsat_folder": {"type": "string", "description": "PLC校正后Landsat SR文件夹"},
                    "model_path": {"type": "string", "description": "训练好的RF模型路径(.mat)"},
                    "output_folder": {"type": "string", "description": "晴空NDVI输出文件夹"}
                },
                "required": ["landsat_folder", "model_path", "output_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reconstruct_ndvi",
            "description": "NDVI时序重建：基于线性回归加权重建云覆盖缺失像元，生成连续8天时序NDVI产品",
            "parameters": {
                "type": "object",
                "properties": {
                    "landsat_folder": {"type": "string", "description": "RF估算后的Landsat NDVI文件夹"},
                    "glass_folder": {"type": "string", "description": "预处理后GLASS NDVI文件夹"},
                    "output_folder": {"type": "string", "description": "重建结果输出文件夹"},
                    "parameter_folder": {"type": "string", "description": "回归参数矩阵保存文件夹"}
                },
                "required": ["landsat_folder", "glass_folder", "output_folder", "parameter_folder"]
            }
        }
    }
]


# ============ 工具分发 ============
def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """
    根据工具名分发到对应处理函数

    Args:
        tool_name: 工具名称
        tool_args: 工具参数字典

    Returns:
        工具执行结果JSON字符串
    """
    print(f"\n[执行工具] {tool_name}")
    print(f"[参数] {json.dumps(tool_args, ensure_ascii=False, indent=2)}")

    if tool_name == "preprocess_glass_ndvi":
        return preprocess_glass_ndvi(**tool_args)

    elif tool_name == "run_plc_correction":
        return run_plc_correction(**tool_args, matlab_script_dir=MATLAB_SCRIPT_DIR)

    elif tool_name == "rf_train":
        return rf_train(**tool_args, matlab_script_dir=MATLAB_SCRIPT_DIR)

    elif tool_name == "rf_apply":
        return rf_apply(**tool_args, matlab_script_dir=MATLAB_SCRIPT_DIR)

    elif tool_name == "reconstruct_ndvi":
        return reconstruct_ndvi(**tool_args, matlab_script_dir=MATLAB_SCRIPT_DIR)

    else:
        return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)


# ============ Agent 系统提示词 ============
TOOL_SYSTEM_PROMPT = """你是一个遥感数据自动化处理助手，专门处理 Landsat NDVI 产品生产任务。
你有以下工具可以调用：

1. preprocess_glass_ndvi：GLASS NDVI预处理（HDF→TIFF→重投影→重采样→裁剪）
2. run_plc_correction：Landsat SR 地形辐射校正
3. rf_train：随机森林模型训练
4. rf_apply：随机森林模型应用，估算晴空NDVI
5. reconstruct_ndvi：NDVI时序重建，填补云覆盖缺失

根据用户需求，判断需要执行哪些步骤，按顺序调用工具完成任务。
每步完成后告知用户进度，全部完成后给出最终汇总。

注意事项：
- 如果需要多步骤处理（如完整的NDVI生产），应逐步调用工具，上一步的输出文件夹作为下一步的输入。
- 如果用户缺少必要参数（如文件夹路径、年份范围），请主动询问用户。
- 工具执行结果会以JSON格式返回，请将其转换为用户友好的说明。"""

CHAT_SYSTEM_PROMPT = """你是一个遥感领域的专业助手，可以帮助用户解答以下问题：
- 遥感基础概念和原理（NDVI、地形校正、随机森林、时序重建等）
- 遥感数据处理方法和流程
- 遥感和测绘领域的技术问题
- 本系统的功能和使用方法

请用专业但易懂的语言回答，适当举例帮助理解。如果用户询问需要执行具体数据处理任务，
请提醒用户提供具体参数（如数据路径、年份范围等）以便调用工具执行。"""


# ============ 工具调用循环 ============
def _run_tool_loop(user_input: str, client: OpenAI) -> str:
    """执行遥感工具调用循环（用于 remote_sensing_processing 意图）"""
    print(f"\n[路由] → 遥感数据处理")

    messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    final_answer = ""

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # 任务完成，LLM 直接回复
        if finish_reason == "stop":
            final_answer = msg.content
            print(f"\n🤖 Agent回复:\n{final_answer}")
            break

        # 有工具调用
        if finish_reason == "tool_calls":
            messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # 执行工具
                tool_result = dispatch_tool(tool_name, tool_args)
                preview = tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                print(f"[工具结果] {preview}")

                # 结果返回给 LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

    return final_answer


def _run_chat(user_input: str, client: OpenAI) -> str:
    """执行闲聊/知识问答（用于 chat 意图）"""
    print(f"[路由] → 对话/知识问答")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        max_tokens=1000
    )

    answer = response.choices[0].message.content
    print(f"\n🤖 Agent回复:\n{answer}")
    return answer


def _handle_unknown(intent_result: IntentResult) -> str:
    """处理未知意图"""
    print(f"[路由] → 未知意图 (置信度: {intent_result.confidence})")
    msg = intent_result.clarification_question or "抱歉，我没能理解您的需求。可以更具体地描述一下您想做什么吗？"
    print(f"\n🤖 Agent回复:\n{msg}")
    return msg


# ============ Agent 主入口 ============
def run_agent(user_input: str, client: OpenAI = None) -> str:
    """
    Agent 对话入口：意图识别 → 路由 → 执行 → 返回结果

    支持的路由：
    - remote_sensing_processing → 工具调用循环
    - standard_qa → RAG 标准查询（需在对话中集成 rag_query）
    - paper_search → 暂不支持提示
    - chat → 对话/知识问答
    - unknown → 引导用户澄清

    Args:
        user_input: 用户自然语言输入
        client: OpenAI 客户端实例（可选，默认创建新实例）

    Returns:
        Agent 最终回复内容
    """
    if client is None:
        client = _create_client()

    print(f"\n{'='*60}")
    print(f"👤 用户: {user_input}")
    print(f"{'='*60}")

    # Step 1: 意图识别
    classifier = IntentClassifier(client, MODEL)
    intent_result = classifier.classify(user_input)

    print(f"[意图] {intent_result.intent} (置信度: {intent_result.confidence:.2f})")

    if intent_result.missing_slots:
        print(f"[缺参] {intent_result.missing_slots}")

    # Step 2: 路由分发
    route = classifier.route(intent_result)

    if route == "tool":
        # 遥感数据处理：进入工具调用循环
        return _run_tool_loop(user_input, client)

    elif route == "rag":
        # 标准查询：提示使用 RAG
        print(f"[路由] → 标准查询 (RAG)")
        answer = (
            "我识别到您想查询遥感/测绘相关标准规范。\n\n"
            "目前知识库中已收录15份国家遥感测绘标准，包括无人机航测、\n"
            "数字测绘成果质量检查、遥感影像处理等规范。\n\n"
            "您可以输入具体问题，我会从标准文档中检索相关内容并给出带来源的答案。\n"
            "例如：\n"
            "- \"无人机航测的航向重叠度要求是多少？\"\n"
            "- \"数字测绘成果质量检查的主要内容有哪些？\""
        )
        print(f"\n🤖 Agent回复:\n{answer}")
        return answer

    elif route == "paper":
        # 论文检索：暂未实现
        print(f"[路由] → 论文检索 (暂不支持)")
        answer = (
            "论文检索功能正在开发中，敬请期待。\n\n"
            "当前我可以帮您：\n"
            "- 处理遥感数据（GLASS NDVI预处理、地形校正、RF训练等）\n"
            "- 查询遥感测绘国家标准和技术规范\n"
            "- 解答遥感领域基础概念和原理"
        )
        print(f"\n🤖 Agent回复:\n{answer}")
        return answer

    elif route == "chat":
        return _run_chat(user_input, client)

    else:
        return _handle_unknown(intent_result)


# ============ 入口 ============
if __name__ == "__main__":
    # 检查 API Key
    if not API_KEY:
        print("⚠️  警告: 未设置 DEEPSEEK_API_KEY 环境变量")
        print("   请创建 .env 文件（参考 .env.example），或设置环境变量后重试。")
        sys.exit(1)

    client = _create_client()

    # 测试不同的意图类型
    test_inputs = [
        "你好，请问这个系统能做什么？",
        "帮我预处理2020年的GLASS NDVI数据",
        "无人机航测的重叠度有什么要求？",
        "帮我搜索NDVI重建的最新论文",
        "asdfghjkl什么都不是的输入",
    ]

    for inp in test_inputs:
        run_agent(inp, client=client)
        print()  # 空行分隔
