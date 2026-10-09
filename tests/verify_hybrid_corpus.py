"""Phase 2 实装验收：混合检索 vs 纯 dense 的召回对比（观察型，无硬断言）。

同一批 query 分别走 KnowledgeRetriever.search（dense 单路）与
hybrid_search（dense + BM25，RRF 融合），并排打印观察。

预期：
- 语义型 query：两路召回基本一致（dense 本就是主力）
- 精确词型 query（型号/缩写/专名）：hybrid 凭 BM25 路提升命中稳定性
- 两种 score 量纲不同（COSINE vs RRF），只比排序不比数值

需要先重跑 build_kb_index.py（hybrid 的 schema 与旧 dense 索引不兼容）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings
from app.agent.rag.embedder import Embedder
from app.agent.rag.retriever import KnowledgeRetriever
from app.agent.rag.milvus_backend import MilvusBackend

_uri = settings.milvus_uri
if not _uri.startswith("http"):
    _uri = str(ROOT / _uri)

backend = MilvusBackend(uri=_uri, collection_name=settings.milvus_collection)
embedder = Embedder(
    api_key=settings.effective_embedding_api_key,
    base_url=settings.effective_embedding_base_url,
    model=settings.embedding_model,
)
r = KnowledgeRetriever(embedder=embedder, backend=backend)

# 语义型：靠向量语义就该命中；精确词型：靠 BM25 词面匹配兜底
queries = [
    ("OLED 和 IPS 屏幕有什么区别", "语义"),
    ("写代码的笔记本需要什么配置", "语义"),
    ("Mini-LED 屏幕怎么样", "精确词"),
    ("ANC 主动降噪原理", "精确词"),
    ("积分怎么获取", "精确词"),
]

for q, kind in queries:
    dense = r.search(q, top_k=3)
    hybrid = r.hybrid_search(q, top_k=3)

    print(f"\n{'=' * 60}\n[{kind}] {q}")
    print("  dense 单路（COSINE）：")
    for hit in dense:
        print(f"    {hit.score:.4f}  {hit.chunk.doc} · {hit.chunk.section}")
    print("  hybrid（dense + BM25 · RRF）：")
    for hit in hybrid:
        print(f"    {hit.score:.4f}  {hit.chunk.doc} · {hit.chunk.section}")
