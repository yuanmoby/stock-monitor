"""
RAG 检索增强生成 Demo — 财报问答系统
用最少的代码展示 RAG 完整链路：分块 → 向量化 → 检索 → 增强生成
作者：齐北
日期：2025.06 初版 / 2026.08 修正

运行前安装依赖：
    pip install sentence-transformers chromadb
首次运行会自动下载中文 Embedding 模型（约100MB）。
如果下载慢，先设置镜像：
    set HF_ENDPOINT=https://hf-mirror.com
"""

import sys

# Windows 中文控制台默认 GBK 编码，打印 emoji 会报 UnicodeEncodeError，
# 统一把标准输出切到 UTF-8（Win11 终端默认支持 UTF-8 显示）
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

print("=" * 60)
print("  RAG 检索增强生成 Demo — 财报问答")
print("=" * 60)

# ============================================================
# 第一步：准备"知识库" — 模拟几份财报摘要
# ============================================================
# 真实场景中这些来自PDF/网页/数据库，这里用文本模拟
documents = [
    "贵州茅台2025年第一季度实现营业收入456.78亿元，同比增长15.2%；归属于上市公司股东的净利润为231.45亿元，同比增长17.8%。",
    "贵州茅台2025年Q1销售毛利率为92.1%，较去年同期的91.8%提升0.3个百分点，主要得益于直销渠道占比提升。",
    "贵州茅台2025年Q1经营活动现金流净额为189亿元，同比下降8.3%，主要原因是预收账款减少。",
    "宁德时代2025年Q1实现营业收入1050亿元，同比增长12.5%；归母净利润为145亿元，同比增长18.2%。",
    "宁德时代2025年Q1动力电池全球市占率达到37.5%，较去年提升2个百分点，连续6年位居全球第一。",
    "比亚迪2025年Q1实现营业收入1800亿元，同比增长20.1%；新能源汽车销量达到95万辆，同比增长22%。",
]

print(f"\n📚 知识库已加载 {len(documents)} 篇文档")

# ============================================================
# 第二步：Embedding — 把文本变成向量
# ============================================================
print("\n🔄 正在将文档转为向量（首次运行会下载模型）...")
# BGE 是智源研究院开源的中文 Embedding 模型，对中文语义效果好且免费可商用
embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 所有文档一次性编码：每行是一个文档的 512 维向量
doc_embeddings = embedder.encode(documents)
print(f"   文档向量矩阵形状：{doc_embeddings.shape}  ← ({len(documents)}篇文档, 512维)")

# ============================================================
# 第三步：存入向量数据库（ChromaDB，开源免费）
# ============================================================
chroma_client = chromadb.Client()
# hnsw:space 指定用余弦距离，与 BGE 模型的训练目标一致（向量方向比长度重要）
collection = chroma_client.create_collection(
    name="financial_reports",
    metadata={"hnsw:space": "cosine"},
)

for i, doc in enumerate(documents):
    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[doc_embeddings[i].tolist()],
        documents=[doc],
    )
print("   文档已存入 ChromaDB 向量数据库（余弦空间）")

# ============================================================
# 第四步：检索 — 用户提问，找最相关的文档
# ============================================================

def retrieve(query: str, top_k: int = 2):
    """把问题也编码成向量，在库里做相似度搜索，返回 top_k 篇文档"""
    query_embedding = embedder.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )

    retrieved_docs = results["documents"][0]
    distances = results["distances"][0]

    # 余弦空间下：distance = 1 - 余弦相似度，所以相似度 = 1 - distance
    similarities = [1 - d for d in distances]
    return retrieved_docs, similarities


# ---- 测试检索 ----
test_queries = [
    "茅台一季度赚了多少钱？",
    "宁德时代的市场份额是多少？",
    "比亚迪卖了多少辆车？",
    "茅台的毛利率有什么变化？",
]

for query in test_queries:
    print(f"\n{'─' * 50}")
    print(f"🙋 用户问题：{query}")

    docs, sims = retrieve(query, top_k=2)

    for i, (doc, sim) in enumerate(zip(docs, sims)):
        print(f"   📄 检索结果{i+1}（余弦相似度 {sim:.2%}）：")
        print(f"      {doc[:80]}...")

# ============================================================
# 第五步：增强生成（Demo 中模拟 LLM 回复）
# ============================================================
# 实际接入 LLM API 只需几行，例如 DeepSeek：
#
#   from openai import OpenAI
#   client = OpenAI(api_key="你的key", base_url="https://api.deepseek.com")
#   response = client.chat.completions.create(
#       model="deepseek-chat",
#       messages=[{"role": "user", "content": prompt}],
#   )
#   print(response.choices[0].message.content)

print(f"\n{'=' * 60}")
print("  🧠 增强生成（模拟LLM回复）")
print("=" * 60)

query = "茅台一季度净利润是多少？毛利率怎么样？"
retrieved_docs, _ = retrieve(query, top_k=2)

# 构建增强 Prompt：把检索到的资料塞进提示词，让 LLM 基于真实数据回答
context = "\n".join([f"- {doc}" for doc in retrieved_docs])
prompt = f"""你是一个金融分析助手。请严格根据以下资料回答问题。
如果资料中没有相关信息，请明确说"资料中未提及"。

【参考资料】
{context}

【用户问题】
{query}

【回答】"""

print(f"\n📝 构建的增强 Prompt：")
print(f"{prompt[:400]}...")

print(f"\n📋 模拟回答（基于检索到的上下文）：")
print(f"   根据贵州茅台2025年Q1财报：")
print(f"   1. 归母净利润为231.45亿元，同比增长17.8%")
print(f"   2. 销售毛利率为92.1%，较去年同期提升0.3个百分点")
print(f"   3. 提升原因为直销渠道占比提升")

print(f"\n{'=' * 60}")
print(f"✅ RAG Demo 运行完毕！")
print(f"   核心链路：文档分块 → Embedding向量化 → 向量检索 → 增强Prompt → LLM生成")
print(f"   下一步：接入真实LLM API（DeepSeek/通义千问有免费额度）")
