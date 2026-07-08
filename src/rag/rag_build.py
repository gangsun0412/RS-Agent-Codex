"""
RAG 知识库构建工具
读取遥感测绘标准 PDF，切片后向量化存入 Chroma 数据库
"""
import os
import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdfs(pdf_folder: str, file_list: list) -> list:
    """读取指定PDF文件列表，返回包含内容和来源的文档列表"""
    documents = []
    for pdf_file in file_list:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            if full_text.strip():
                documents.append({
                    "content": full_text,
                    "source": pdf_file
                })
                print(f"✅ 读取成功: {pdf_file} ({len(full_text)}字符)")
            else:
                print(f"⚠️ 内容为空: {pdf_file}")
        except Exception as e:
            print(f"❌ 读取失败: {pdf_file} ({e})")
    return documents


def split_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """将文档切分为固定大小的文本块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for split in splits:
            chunks.append({
                "content": split,
                "source": doc["source"]
            })
    print(f"\n共切分为 {len(chunks)} 个文本块")
    return chunks


def build_vector_db(chunks: list, vector_db_path: str) -> Chroma:
    """将文本块向量化并存入Chroma数据库"""
    print("\n加载Embedding模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]

    print("开始向量化并写入数据库...")
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=vector_db_path
    )
    print(f"✅ 向量数据库构建完成，保存至: {vector_db_path}")
    return vectordb


# ============ 主流程 ============
if __name__ == "__main__":
    # 配置路径（可根据实际环境修改）
    PDF_FOLDER = r"E:\遥感-无人机-测绘标准"
    VECTOR_DB_PATH = r"D:\RS_Agent_Codex\vector_db"

    # 15份正常标准PDF文件列表
    normal_files = [
        '18316-2008-gbt-e-300.pdf',
        '27919-2011-gbt-e-300.pdf',
        '27920_1-2011-gbt-e-300.pdf',
        '27920_2-2012-gbt-e-300.pdf',
        '7930-2008-gbt-e-300.pdf',
        '7931-2008-gbt-e-300.pdf',
        'GBT+23236-2024.pdf',
        'GBT+40527-2021.pdf',
        'GBT+40766-2021.pdf',
        'GBT+41149-2021.pdf',
        'GBT+41449-2022.pdf',
        'GBT+41450-2022.pdf',
        'GBT+43551-2023.pdf',
        'GBT+43995-2024.pdf',
        'GBT+45522-2025.pdf',
    ]

    print("===== Step1: 读取PDF =====")
    documents = load_pdfs(PDF_FOLDER, normal_files)

    print("\n===== Step2: 文本切片 =====")
    chunks = split_documents(documents)

    print("\n===== Step3: 构建向量数据库 =====")
    vectordb = build_vector_db(chunks, VECTOR_DB_PATH)
