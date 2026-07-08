#!/usr/bin/env python
"""
RS Agent Codex - 遥感自动化生产智能体
命令行交互入口
"""
import os
import sys

# 将项目根目录加入路径
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # 尝试 .env.example 作为回退（仅用于提示，不含真实key）
        print("💡 提示: 未找到 .env 文件，请复制 .env.example 为 .env 并填入配置。")
except ImportError:
    pass  # python-dotenv 未安装时忽略

from openai import OpenAI
from src.agent.agent_core import (
    API_KEY, BASE_URL, MODEL, MATLAB_SCRIPT_DIR,
    _create_client, run_agent
)
from src.agent.task_logger import TaskLogger


def check_environment() -> dict:
    """检查运行环境，返回状态字典"""
    status = {
        "api_key": False,
        "matlab_scripts": False,
        "matlab_engine": False,
        "vector_db": False,
    }

    # 检查 API Key
    if API_KEY:
        status["api_key"] = True

    # 检查 MATLAB 脚本目录
    if os.path.isdir(MATLAB_SCRIPT_DIR):
        m_files = [f for f in os.listdir(MATLAB_SCRIPT_DIR) if f.endswith(".m")]
        if len(m_files) >= 4:
            status["matlab_scripts"] = True

    # 检查 MATLAB 引擎
    try:
        import matlab.engine
        status["matlab_engine"] = True
    except ImportError:
        pass

    # 检查向量数据库
    vector_db_path = os.path.join(_PROJECT_ROOT, "vector_db")
    if os.path.isdir(vector_db_path) and os.path.exists(
        os.path.join(vector_db_path, "chroma.sqlite3")
    ):
        status["vector_db"] = True

    return status


def print_banner():
    """打印欢迎信息"""
    print(r"""
╔══════════════════════════════════════════════════╗
║           🌏 RS Agent Codex                       ║
║           遥感自动化生产智能体                      ║
║           v0.1.0 - MVP                           ║
╚══════════════════════════════════════════════════╝
""")
    print("可用功能：")
    print("  🔧 遥感数据处理 - GLASS预处理、PLC校正、RF训练/预测、NDVI重建")
    print("  📋 标准规范查询 - 15份国家遥感测绘标准 (RAG)")
    print("  💬 遥感知识问答 - 基础概念和原理")
    print("")
    print("输入 'help' 查看更多, 'quit' 或 'exit' 退出")
    print("-" * 50)


def print_help():
    """打印帮助信息"""
    print("""
📖 使用帮助
═══════════════════════════════════════════════════

【数据处理示例】
   "帮我预处理2017-2018年的GLASS NDVI数据，
    HDF在D:\\data\\hdf，ROI文件D:\\data\\roi.shp，
    输出到D:\\data\\glass_output"

   "对Landsat SR做地形校正，
    影像在D:\\data\\sr，坡度D:\\data\\slope.tif，
    坡向D:\\data\\aspect.tif，输出到D:\\data\\plc"

【标准查询示例】
   "无人机航测的航向重叠度要求是多少？"
   "数字测绘成果质量检查有哪些内容？"

【知识问答示例】
   "什么是NDVI？"
   "随机森林在遥感中有什么应用？"

【命令】
   help   - 显示此帮助
   clear  - 清屏
   status - 查看环境状态
   quit   - 退出

【提示】
   - 数据处理任务需提供具体路径和参数
   - 缺少参数时Agent会主动追问
   - 论文检索功能正在开发中
""")


def print_status(env_status: dict):
    """打印环境状态"""
    print("\n🔍 环境状态检查")
    print("-" * 30)
    items = [
        ("DeepSeek API Key", env_status["api_key"]),
        ("MATLAB 脚本目录", env_status["matlab_scripts"]),
        ("MATLAB 引擎", env_status["matlab_engine"]),
        ("Chroma 向量数据库", env_status["vector_db"]),
    ]
    for name, ok in items:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")

    print(f"\n  📁 MATLAB脚本: {MATLAB_SCRIPT_DIR}")
    print(f"  🌐 API地址: {BASE_URL}")
    print(f"  🤖 模型: {MODEL}")
    print()


def main():
    """主入口：交互式对话循环"""
    print_banner()

    # 环境检查
    env_status = check_environment()
    issues = [k for k, v in env_status.items() if not v]
    if issues:
        print("⚠️  部分依赖未就绪（不影响知识问答功能）：")
        if not env_status["api_key"]:
            print("   ❌ API Key 未配置 - 无法调用 LLM")
        if not env_status["matlab_engine"]:
            print("   ⚠️  MATLAB 引擎未安装 - 无法执行遥感处理")
        if not env_status["matlab_scripts"]:
            print("   ⚠️  MATLAB 脚本未找到 - 请检查 matlab_scripts/ 目录")
        if not env_status["vector_db"]:
            print("   ⚠️  Chroma 向量库未找到 - 标准查询不可用，请先运行 rag_build.py")
        print()

    if not env_status["api_key"]:
        print("⛔ 缺少 API Key，请配置 .env 文件后重启。")
        return

    # 初始化客户端
    try:
        client = _create_client()
    except RuntimeError as e:
        print(f"⛔ {e}")
        return

    # 初始化日志
    log_dir = os.path.join(_PROJECT_ROOT, "workspace", "logs")
    task_logger = TaskLogger(log_dir)

    # 对话循环
    print("👋 请输入您的问题或任务：\n")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # 处理内置命令
        cmd = user_input.lower()
        if cmd in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        elif cmd == "help":
            print_help()
            continue
        elif cmd == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            continue
        elif cmd == "status":
            env_status = check_environment()
            print_status(env_status)
            continue
        elif cmd in ("",):
            continue

        # 记录任务开始
        task_id = task_logger.start_task(user_input)
        print()  # 空行美化

        try:
            final_answer = run_agent(user_input, client=client)
            task_logger.finish_task(
                status="success",
                final_answer=final_answer
            )
        except Exception as e:
            print(f"\n❌ 执行异常: {e}")
            task_logger.finish_task(
                status="failed",
                error=str(e)
            )

        print()  # 空行分隔


if __name__ == "__main__":
    main()
