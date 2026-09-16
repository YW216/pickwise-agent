"""临时验证：RAG 选购语料召回（跑完可删）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings
from app.agent.rag.milvus_backend import MilvusBackend
from app.agent.rag.embedder import Embedder
from app.agent.rag.retriever import KnowledgeRetriever

# lite（本地相对路径）按项目根解析；standalone（http://）原样透传
_uri = settings.milvus_uri
if not _uri.startswith("http"):
    _uri = str(ROOT / _uri)

backend = MilvusBackend(
    uri=_uri,
    collection_name=settings.milvus_collection,
)
embedder = Embedder(
    api_key=settings.embedding_api_key,
    base_url=settings.embedding_base_url,
    model=settings.embedding_model,
)
r = KnowledgeRetriever(embedder=embedder, backend=backend)

# 典型查询：前两个是实测暴露的知识缺口
queries = [
    ("OLED 和 IPS 屏幕有什么区别", "屏幕面板科普"),
    ("写代码的笔记本需要什么配置", "笔记本选购指南"),
    ("耳机怎么选降噪", "耳机选购指南"),
    ("手机拍照好不好看什么", "手机选购指南"),
]

for q, expect_doc in queries:
    results = r.search(q, top_k=3)
    docs = [res.chunk.doc for res in results]
    if expect_doc in docs:
        top = results[0]
        print(f"✅ 「{q}」→ 命中《{expect_doc}》 | top1: {top.chunk.doc} | {top.chunk.text[:40]}...")
    else:
        print(f"❌ 「{q}」→ 未命中《{expect_doc}》，返回: {docs}")
