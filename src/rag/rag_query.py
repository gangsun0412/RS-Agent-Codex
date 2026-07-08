"""
RAG 知识库查询工具
检索遥感测绘标准，基于检索结果用 LLM 生成带来源引用的专业回答
"""
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI


def load_vector_db(vector_db_path: str) -> Chroma:
    """加载Chroma向量数据库"""
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectordb = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embedding_model
    )
    print("✅ 向量数据库加载成功")
    return vectordb


def retrieve(vectordb: Chroma, query: str, top_k: int = 5) -> list:
    """检索与查询最相关的文档块"""
    results = vectordb.similarity_search(query, k=top_k)
    return results


def rag_query(
    vectordb: Chroma,
    question: str,
    client: OpenAI,
    model: str = "deepseek-chat"
) -> str:
    """
    RAG查询：检索相关标准文档，用LLM生成带来源的专业回答

    Args:
        vectordb: Chroma向量数据库实例
        question: 用户问题
        client: OpenAI客户端实例
        model: LLM模型名

    Returns:
        LLM生成的回答字符串，含来源引用；检索无结果时返回拒答提示
    """
    print(f"\n问题: {question}")
    print("-" * 50)

    # Step1: 检索相关文档块
    docs = retrieve(vectordb, question)
    if not docs:
        return "未检索到相关内容，知识库中暂无与该问题相关的标准依据。"

    # Step2: 拼接检索结果
    context = ""
    sources = set()
    for i, doc in enumerate(docs):
        context += f"【片段{i + 1}】来源：{doc.metadata.get('source', '未知')}\n"
        context += f"{doc.page_content}\n\n"
        sources.add(doc.metadata.get('source', '未知'))

    print(f"检索到 {len(docs)} 个相关片段，来源：{sources}")

    # Step3: 构造提示词
    prompt = f"""你是一个遥感测绘领域的专业助手，请根据以下检索到的标准文档内容回答用户问题。
回答要准确、专业，并注明依据来自哪份标准文档。
如果检索内容中没有相关信息，请如实说明"知识库中未找到可靠依据"。

检索到的相关内容：
{context}

用户问题：{question}

请基于以上内容给出专业回答："""

    # Step4: 调用LLM生成回答
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    answer = response.choices[0].message.content
    return answer


def create_rag_client(
    vector_db_path: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat"
) -> dict:
    """
    创建RAG查询所需的客户端实例

    Args:
        vector_db_path: 向量数据库路径
        api_key: DeepSeek API密钥
        base_url: API地址
        model: 模型名

    Returns:
        包含vectordb、client、model的字典
    """
    vectordb = load_vector_db(vector_db_path)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return {
        "vectordb": vectordb,
        "client": client,
        "model": model
    }


# ============ 主程序 ============
if __name__ == "__main__":
    VECTOR_DB_PATH = r"D:\RS_Agent_Codex\vector_db"

    # 从环境变量读取API key
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    rag = create_rag_client(VECTOR_DB_PATH, API_KEY, BASE_URL)

    questions = [
        "无人机航测的航向重叠度要求是多少？",
        "数字测绘成果质量检查的主要内容有哪些？",
        "遥感影像的分辨率如何分类？"
    ]

    for q in questions:
        answer = rag_query(rag["vectordb"], q, rag["client"], rag["model"])
        print(f"\n回答:\n{answer}")
        print("=" * 60)
