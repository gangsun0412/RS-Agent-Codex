# 遥感自动化生产 Agent 开发设计文档 v2

## 1. 文档目标

本文档用于指导遥感领域智能体（Agent）的后续开发，将当前已有的脚本工具、RAG 知识库和计划中的交互界面整合为一个可维护、可扩展、可验收的本地遥感自动化生产系统。

当前原始文档已经记录了项目背景、已完成模块和下一步优先级。本版本在此基础上补充产品定位、系统边界、模块职责、任务流程、输入输出契约、异常处理、测试验收和理想状态下应具备的关键能力。

## 2. 项目定位

### 2.1 Agent 目标

开发一个面向遥感生产与遥感知识服务的本地智能体，使用户可以通过自然语言完成以下任务：

- 调用遥感处理工具，完成 GLASS NDVI 预处理、Landsat 地形校正、随机森林 NDVI 估算、NDVI 时序重建等流程。
- 查询遥感、测绘、无人机等相关国家标准或技术规范。
- 检索遥感领域论文，并对结果进行摘要、筛选和引用整理。
- 在多轮对话中管理任务参数、执行状态、输出文件和后续操作建议。

### 2.2 目标用户

主要用户包括：

- 遥感科研人员。
- 遥感工程项目开发人员。
- 测绘、生态、农业、自然资源等领域的数据处理人员。
- 需要快速查询遥感标准和论文资料的研究生或技术人员。

### 2.3 理想状态

理想状态下，该 Agent 不只是一个调用脚本的聊天壳，而是一个具备任务理解、参数补全、流程编排、异常诊断、结果解释和知识查询能力的遥感工作助手。

用户可以输入类似请求：

- “帮我处理 2020 到 2022 年研究区的 GLASS NDVI，并裁剪到这个 ROI。”
- “用 Landsat SR 和 GLASS NDVI 训练随机森林模型，然后估算晴空 NDVI。”
- “查询无人机航测成果质量检查相关标准。”
- “检索近三年 NDVI 时序重建和随机森林融合相关论文。”

Agent 应能判断任务类型，追问缺失参数，调用对应工具，展示执行进度，保存结果，并给出可追溯的输出说明。

## 3. 当前已有基础

### 3.1 环境配置

- Python 3.11 虚拟环境。
- DeepSeek API，兼容 OpenAI 调用格式。
- Matlab 2023b，本地 `matlab.engine` 调用已测试通过。
- 已安装主要依赖：`openai`、`matlab.engine`、`rasterio`、`gdal`、`numpy`、`langchain`、`chromadb`、`sentence-transformers`、`pymupdf`、`langchain-huggingface`、`langchain-text-splitters`。

### 3.2 已有遥感工具

#### 3.2.1 GLASS NDVI 预处理

文件：

- `tool_preprocess.py`

功能：

- HDF 转 TIFF。
- 重投影到 EPSG:4326。
- 重采样到 240m。
- 按 ROI 裁剪。

主要输入：

- HDF 文件夹。
- 输出文件夹。
- ROI shapefile。
- 起止年份。

主要输出：

- 裁剪后的 TIFF 文件。
- 命名格式：`yyyymmdd.tif`。

#### 3.2.2 PLC 地形校正

文件：

- `tool_plc.py`
- `plc_correction.m`

功能：

- 对 Landsat SR 影像进行地形辐射校正。
- 输入 8 波段，输出前 7 波段。

主要输入：

- Landsat SR 影像文件夹。
- `slope.tif`。
- `aspect.tif`。
- 输出文件夹。

#### 3.2.3 随机森林训练与应用

文件：

- `tool_rf.py`
- `tool_rf_train.m`
- `tool_rf_apply.m`

功能：

- 使用 Landsat SR 7 波段和 GLASS NDVI 训练随机森林模型。
- 使用训练好的模型估算晴空 NDVI。

主要输出：

- `.mat` 模型文件。
- `[-1, 1]` 范围的 double 单波段 NDVI TIFF。

#### 3.2.4 NDVI 时序重建

文件：

- `tool_reconstruct.py`
- `tool_reconstruct.m`

功能：

- 第一阶段：16 天 NDVI 重建。
- 第二阶段：补充为 8 天时序。

主要输入：

- RF 估算 NDVI 文件夹。
- GLASS 文件夹。
- 输出文件夹。
- 参数矩阵文件夹。

主要约定：

- nodata 统一为 `NaN`。

### 3.3 已有 Agent 主框架

文件：

- `agent_core.py`

已完成能力：

- 使用 DeepSeek API。
- 注册四个遥感处理 Tool。
- 通过 `dispatch_tool` 分发工具调用。
- 支持“用户输入 -> LLM 决策 -> 工具调用 -> 返回结果 -> 继续或结束”的基础循环。

### 3.4 已有 RAG 知识库

文件：

- `rag_build.py`
- `rag_query.py`

已完成能力：

- 读取 15 份国家遥感测绘标准 PDF。
- 使用 BGE 中文 embedding 模型。
- 构建 Chroma 向量数据库。
- 查询 top5 片段，并将 context 交给 DeepSeek 生成回答。
- 回答中自动标注来源文档。

## 4. 当前设计存在的主要问题

### 4.1 项目目标不够具体

原始文档说明了“面向遥感垂直领域”，但没有明确最终服务对象、核心场景和验收标准。后续需要明确 Agent 是偏自动化生产、标准问答、论文助手，还是三者统一入口。

### 4.2 Agent 与脚本工具边界不清

当前文档详细列出了工具脚本，但没有说明 Agent 具体负责哪些智能决策，例如：

- 如何选择工具。
- 如何补全参数。
- 如何组织多工具流程。
- 工具失败后如何处理。
- 如何向用户解释执行结果。

### 4.3 端到端数据链路尚未定义

四个遥感工具之间存在明显的上下游关系，但文档尚未明确完整数据流。例如：

- GLASS 预处理结果如何进入 RF 训练。
- PLC 校正后的 Landsat SR 如何作为 RF 输入。
- RF 估算 NDVI 与 GLASS NDVI 如何进入时序重建。
- 时间分辨率、空间分辨率、投影和 ROI 如何统一。

### 4.4 工具输入输出契约不足

每个 Tool 目前只描述了大致输入输出，缺少稳定工程接口所需的细节：

- 参数类型。
- 必填字段。
- 默认值。
- 文件命名规则。
- 支持格式。
- 坐标系要求。
- nodata 规则。
- 返回值结构。
- 错误码或错误信息格式。

### 4.5 意图识别设计过于简略

原始文档提出让 LLM 输出 JSON 意图标签，但还缺少：

- 意图枚举。
- JSON schema。
- 置信度字段。
- 参数槽位抽取。
- 缺参追问机制。
- 多意图处理。
- 路由失败时的回退策略。

### 4.6 RAG 查询缺少质量控制

标准查询属于严肃知识问答，需要补充：

- 来源页码。
- 标准名称和版本。
- 检索为空时的拒答策略。
- 多标准冲突时的说明方式。
- 引用格式。
- 知识库更新机制。

### 4.7 缺少长任务状态管理

遥感数据处理通常耗时较长，需要明确：

- 任务 ID。
- 当前步骤。
- 进度显示。
- 执行日志。
- 中间结果。
- 失败恢复。
- 是否允许取消任务。
- 是否允许并发任务。

### 4.8 缺少异常处理策略

需要提前定义常见错误场景：

- 文件或目录不存在。
- ROI 与影像不重叠。
- 投影不一致。
- 影像波段数不符合要求。
- Matlab engine 启动失败。
- API key 不可用。
- LLM 返回非法 JSON。
- RAG 检索无结果。
- 输出目录已有同名文件。

### 4.9 缺少测试和验收标准

需要建立可验证的完成标准，包括：

- 每个 Tool 的最小样例测试。
- 端到端 Workflow 测试。
- 意图识别测试集。
- RAG 问答准确性测试。
- 异常场景测试。
- 输出遥感产品质量检查。

## 5. 建议的系统架构

建议将系统拆分为以下层次：

```text
用户界面层
  Streamlit 页面 / 命令行入口 / 后续可扩展 API 服务

Agent 编排层
  对话管理
  意图识别
  参数抽取
  缺参追问
  Workflow 路由
  结果解释

工具执行层
  GLASS 预处理 Tool
  PLC 校正 Tool
  RF 训练与应用 Tool
  NDVI 时序重建 Tool
  RAG 查询 Tool
  论文检索 Tool

资源管理层
  数据目录
  模型目录
  输出目录
  日志目录
  任务状态目录
  向量数据库目录

外部依赖层
  DeepSeek API
  Matlab engine
  Chroma
  arXiv / Semantic Scholar API
```

## 6. 核心模块设计

### 6.1 意图识别模块

目标：

- 判断用户请求属于哪类任务。
- 抽取关键参数。
- 判断参数是否完整。
- 输出结构化 JSON，供路由器使用。

建议意图类型：

- `remote_sensing_processing`：遥感数据处理。
- `standard_qa`：标准规范查询。
- `paper_search`：论文检索。
- `chat`：普通闲聊或解释性问答。
- `unknown`：无法判断。

建议输出结构：

```json
{
  "intent": "remote_sensing_processing",
  "confidence": 0.86,
  "task_name": "ndvi_reconstruction",
  "slots": {
    "roi_path": "",
    "start_year": 2020,
    "end_year": 2022,
    "input_dir": "",
    "output_dir": ""
  },
  "missing_slots": ["roi_path", "input_dir", "output_dir"],
  "need_user_clarification": true,
  "clarification_question": "请提供 ROI 文件路径、输入数据文件夹和输出文件夹。"
}
```

### 6.2 Workflow 编排模块

目标：

- 将单个工具调用和多工具链路统一管理。
- 支持固定流程和动态流程。
- 支持中间结果传递。

建议定义的 Workflow：

- `glass_preprocess_workflow`：GLASS NDVI 预处理。
- `plc_correction_workflow`：Landsat SR 地形校正。
- `rf_train_workflow`：随机森林模型训练。
- `rf_apply_workflow`：晴空 NDVI 估算。
- `ndvi_reconstruction_workflow`：NDVI 时序重建。
- `full_ndvi_production_workflow`：从原始数据到时序重建结果的完整流程。

### 6.3 工具注册与调用模块

每个 Tool 都应具备统一结构：

- `name`：工具名称。
- `description`：工具用途。
- `parameters`：Function Calling 参数 schema。
- `required`：必填参数。
- `run()`：实际执行函数。
- `validate_inputs()`：输入检查。
- `summarize_result()`：执行结果摘要。

工具返回建议统一为：

```json
{
  "success": true,
  "tool_name": "tool_preprocess",
  "message": "GLASS NDVI 预处理完成。",
  "outputs": [
    "outputs/glass_preprocess/20200101.tif"
  ],
  "metadata": {
    "file_count": 46,
    "crs": "EPSG:4326",
    "resolution": "240m"
  },
  "error": null
}
```

### 6.4 RAG 标准查询模块

目标：

- 查询标准规范。
- 返回有来源、有依据、可追溯的回答。

建议增强：

- 检索结果包含文档名、页码、片段文本、相似度分数。
- 回答必须基于检索片段。
- 检索不足时明确说明“知识库中未找到可靠依据”。
- 支持更新或重建知识库。

### 6.5 论文检索模块

目标：

- 检索遥感领域最新论文。
- 对论文结果进行排序、摘要和引用整理。

建议能力：

- 支持中文查询转英文关键词。
- 支持按时间、相关性、引用量排序。
- 输出标题、作者、年份、摘要、链接、DOI。
- 支持生成中文总结。
- 支持导出 BibTeX 或 Markdown 列表。

### 6.6 任务状态与日志模块

目标：

- 让长时间运行的遥感任务可追踪、可恢复、可解释。

建议记录：

- `task_id`。
- 用户原始请求。
- 识别到的 intent。
- 调用的 workflow。
- 每一步工具参数。
- 开始时间、结束时间、耗时。
- 当前状态：`pending`、`running`、`success`、`failed`、`cancelled`。
- 输出文件路径。
- 错误信息。

### 6.7 文件与数据资产管理模块

目标：

- 将代码、数据、模型、输出和日志分类管理。
- 避免硬编码绝对路径。

建议统一从项目根目录派生路径，不将关键路径散落到不同磁盘。

## 7. 建议目录结构

如果把项目代码、数据和必要材料都放在当前目录下，建议采用如下结构：

```text
RS_Agent_Codex/
├── docs/                         # 开发文档、需求文档、使用说明
├── src/                          # Python 源码
│   ├── agent/                    # Agent 主逻辑
│   ├── tools/                    # Function Calling 工具封装
│   ├── workflows/                # 多步骤流程编排
│   ├── rag/                      # RAG 构建与查询
│   ├── literature/               # 论文检索模块
│   └── ui/                       # Streamlit 页面
├── matlab_scripts/               # Matlab 脚本
├── data/                         # 原始数据和外部输入
│   ├── raw/                      # 原始遥感数据
│   ├── roi/                      # ROI 文件
│   ├── dem/                      # slope、aspect、DEM 等辅助数据
│   └── standards/                # 标准 PDF 文档
├── workspace/                    # 任务运行区
│   ├── intermediate/             # 中间结果
│   ├── outputs/                  # 最终输出
│   ├── logs/                     # 日志
│   └── tasks/                    # 任务状态文件
├── models/                       # RF 模型、参数矩阵等
├── vector_db/                    # Chroma 向量数据库
├── tests/                        # 测试代码和小样例数据
├── configs/                      # 配置文件
├── requirements.txt
├── README.md
└── .env.example
```

## 8. 数据放置建议

可以将项目代码、数据和其他必要材料都放在当前目录下，并按类别分文件夹。这种方式非常适合当前阶段开发，因为：

- 后续 Agent 调用路径更统一。
- 更容易迁移到其他机器。
- 更容易做任务日志和输出管理。
- 更方便让 Codex 或其他开发工具理解项目上下文。
- 可以减少 `E:\pythonCode\CodexProject` 与 `D:\Codex_code` 这类跨目录依赖。

需要注意：

- 大体积遥感原始数据不要直接提交到 Git。
- 模型文件、输出结果、临时文件应放入独立目录。
- API key 不要写入代码或文档，建议使用 `.env`。
- 路径尽量通过配置文件统一管理。
- 原始数据目录建议只读，处理结果写入 `workspace/outputs/`。

## 9. 推荐配置文件设计

建议新增 `configs/settings.yaml` 或 `configs/settings.json`，集中管理路径和模型配置。

示例：

```yaml
project:
  root: "D:/RS_Agent_Codex"

paths:
  raw_data: "data/raw"
  roi: "data/roi"
  dem: "data/dem"
  standards: "data/standards"
  outputs: "workspace/outputs"
  intermediate: "workspace/intermediate"
  logs: "workspace/logs"
  tasks: "workspace/tasks"
  models: "models"
  vector_db: "vector_db"

llm:
  provider: "deepseek"
  model: "deepseek-chat"

rag:
  embedding_model: "BAAI/bge-small-zh-v1.5"
  top_k: 5
```

## 10. 异常处理设计

建议将错误分为以下类别：

- `InputPathError`：输入路径不存在或格式不正确。
- `DataValidationError`：数据投影、波段、分辨率、时间范围不符合要求。
- `ToolExecutionError`：工具执行失败。
- `MatlabEngineError`：Matlab engine 启动或运行失败。
- `LLMResponseError`：模型输出格式不合法。
- `RAGNoEvidenceError`：RAG 未检索到可靠依据。
- `ExternalAPIError`：论文检索 API 请求失败。

每个错误都应返回：

- 错误类型。
- 简短说明。
- 可操作的修复建议。
- 原始异常信息。
- 是否可重试。

## 11. 测试与验收

### 11.1 工具级测试

每个遥感 Tool 至少准备一个小样例数据，验证：

- 输入检查是否正确。
- 输出文件是否生成。
- 输出格式是否符合约定。
- 错误输入是否能给出清晰提示。

### 11.2 Agent 路由测试

准备一组自然语言请求，覆盖：

- 数据处理。
- 标准查询。
- 论文检索。
- 闲聊。
- 模糊请求。
- 多意图请求。

验证 Agent 是否能输出正确 intent、slots 和 clarification。

### 11.3 RAG 测试

准备标准问题集，验证：

- 是否检索到正确标准。
- 是否引用来源。
- 是否避免无依据回答。
- 标准版本或文档冲突时是否说明。

### 11.4 端到端测试

使用一套小型遥感样例数据，验证完整流程：

1. GLASS NDVI 预处理。
2. Landsat SR 地形校正。
3. RF 模型训练。
4. RF NDVI 估算。
5. NDVI 时序重建。
6. 输出结果统计和日志记录。

## 12. 后续开发优先级

建议按以下顺序推进：

1. 整理项目目录，将代码、数据、模型、输出和文档分类放置。
2. 抽象统一配置文件，去除硬编码绝对路径。
3. 为现有四个 Tool 补充标准 Function Calling schema。
4. 增加输入检查和统一返回结构。
5. 开发意图识别与参数抽取模块。
6. 开发缺参追问机制。
7. 开发 Workflow 编排模块。
8. 整合 RAG 查询为标准工具。
9. 增加论文检索工具。
10. 增加任务状态、日志和异常处理。
11. 开发 Streamlit 页面。
12. 建立测试样例和验收流程。

## 13. 最小可用版本目标

第一阶段不必一次性完成全部理想能力，建议先实现最小可用版本：

- 支持命令行或简单 Streamlit 对话入口。
- 支持意图识别：数据处理、标准查询、论文检索、闲聊。
- 支持对四个遥感 Tool 的稳定调用。
- 支持缺少路径和年份时主动追问。
- 支持 RAG 标准查询并返回来源。
- 支持任务日志记录。
- 支持输出文件路径和执行摘要返回。

达到这一阶段后，再继续增强可视化、论文总结、任务恢复、质量评估和完整生产链路。

