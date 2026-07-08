# RS Agent Codex — 遥感自动化生产智能体

面向遥感生产与遥感知识服务的本地智能体（Agent），通过自然语言驱动遥感数据处理、标准查询和论文检索。

## 功能概览

- **遥感数据处理**：GLASS NDVI 预处理、Landsat 地形校正、随机森林 NDVI 估算、NDVI 时序重建
- **标准规范查询（RAG）**：检索 15 份国家遥感测绘标准，返回有来源、可追溯的专业回答
- **论文检索**：检索遥感领域论文，支持摘要、筛选和引用整理（开发中）
- **多轮对话**：支持任务参数追问、执行状态展示和结果解释

## 系统架构

```
用户界面层   →  命令行 / Streamlit（待开发）
Agent 编排层 →  对话管理、意图识别、参数抽取、工作流路由
工具执行层   →  GLASS预处理、PLC校正、RF训练/预测、NDVI重建、RAG查询
资源管理层   →  数据目录、模型目录、输出目录、日志、向量数据库
外部依赖层   →  DeepSeek API、Matlab Engine、Chroma
```

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.11+ | 主语言 |
| MATLAB | 2023b+ | 地形校正、RF训练、时序重建 |
| DeepSeek API | - | LLM推理 |
| GDAL | 3.8+ | 遥感数据读写 |

## 快速开始

### 1. 克隆仓库

```bash
git clone <repo-url>
cd RS_Agent_Codex
```

### 2. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装 Python 依赖
pip install -r requirements.txt
```

> **注意**：GDAL 在 Windows 上推荐从 [GIS Internals](https://www.gisinternals.com/) 下载预编译 whl 安装。MATLAB 引擎需从 MATLAB 安装目录手动安装。

### 3. 配置环境变量

```bash
# 复制配置模板
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# 编辑 .env，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-your-key-here
```

### 4. 运行

```bash
# 交互式对话模式
python run.py

# 或直接运行 Agent（测试用）
python -m src.agent.agent_core
```

## 项目结构

```
RS_Agent_Codex/
├── docs/                    # 开发设计文档
├── src/
│   ├── agent/               # Agent 主逻辑（agent_core.py）
│   ├── tools/                # 遥感处理工具（4个Tool）
│   ├── rag/                  # RAG 知识库构建与查询
│   └── ui/                   # Streamlit 界面（待开发）
├── matlab_scripts/           # MATLAB 脚本
├── configs/                  # 配置文件
├── tests/                    # 测试代码
├── data/                     # 数据目录
├── workspace/                # 运行时输出
├── models/                   # 模型文件
├── vector_db/                # Chroma 向量数据库
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板
└── README.md                 # 本文件
```

## 可用工具

| 工具 | 功能 | MATLAB |
|------|------|--------|
| `preprocess_glass_ndvi` | GLASS NDVI 预处理（HDF→TIFF→重投影→重采样→裁剪） | ❌ |
| `run_plc_correction` | Landsat SR PLC 地形辐射校正 | ✅ |
| `rf_train` | 随机森林模型训练 | ✅ |
| `rf_apply` | 随机森林晴空 NDVI 估算 | ✅ |
| `reconstruct_ndvi` | NDVI 时序重建（16天→8天） | ✅ |

## 开发计划

参见 [docs/遥感自动化生产Agent开发设计文档_v2.md](docs/遥感自动化生产Agent开发设计文档_v2.md)

## 许可证

待定
