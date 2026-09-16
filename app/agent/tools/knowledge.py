"""retrieve_knowledge：平台知识库检索（导购 / 比选 / 咨询）。

复用第 5 期 RAG 基础设施（Embedder + MilvusBackend + build_kb_index.py），
检索平台知识库（选购指南、参数科普、售后政策、FAQ、配送、会员权益），返回 Top-K 命中片段及来源。

与结构化查询工具（search_catalog 等）不同，本工具面向**非结构化文本**：
模型知识有限时（如"写代码需要什么配置"）用它查资料补足。

检索器单例缓存，避免每次调用重建索引。
"""

from typing import Optional

from app.agent.rag.embedder import Embedder
from app.agent.rag.milvus_backend import MilvusBackend
from app.agent.rag.retriever import KnowledgeRetriever
from app.agent.tools.result import fail, ok
from app.config.settings import settings

_retriever: Optional[KnowledgeRetriever] = None

# 懒加载单例：首次调用才创建检索器并加载索引。
def _get_retriever() -> KnowledgeRetriever:
    """懒加载单例：首次调用才创建检索器并加载索引。"""
    global _retriever
    if _retriever is None:
        embedder = Embedder(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
        )
        backend = MilvusBackend(
            uri=settings.milvus_uri,
            collection_name=settings.milvus_collection,
        )
        _retriever = KnowledgeRetriever(embedder=embedder, backend=backend)
        _retriever.load()  #连接backend,backend连接数据库，加载索引
    return _retriever


def reset_retriever() -> None:
    """清空单例缓存（测试或切换后端时使用）。"""
    global _retriever
    _retriever = None


def retrieve_knowledge(query: str, top_k: int = 3) -> dict:
    """检索平台知识库（选购指南、参数科普、售后政策、FAQ、配送、会员权益）。

    Args:
        query：用户问题的简洁中文描述（如"写代码需要什么配置"）
        top_k：返回片段数（1-5，默认 3）

    Returns:
        成功：data = {"backend", "query", "results": [{doc, section, score, text}]}
        query 为空：fail("请提供要检索的问题")
        知识库未初始化：fail("知识库未初始化，请先运行 build_kb_index.py")
    """
    if not query or not query.strip():
        return fail("请提供要检索的问题")

    try:
        retriever = _get_retriever()  #查询并创建检索器，加载索引
    except FileNotFoundError:
        # 索引文件不存在——最常见的失败，给出可操作提示
        return fail(
            "知识库未初始化，请先运行 build_kb_index.py",
            {"backend": "milvus", "query": query, "results": []},
        )
    except Exception:
        # 其余异常（模型不一致、服务不可用等）统一人话兜底
        return fail(
            "知识检索失败",
            {"backend": "milvus", "query": query, "results": []},
        )

    top_k = max(1, min(int(top_k or 3), 5))
    # 2026-09-13 切换：dense 单路 → hybrid（dense + BM25，服务端 RRF 融合）。
    # 注意 score 量纲变为 RRF 值（约 0.016~0.033），与旧 COSINE 分不可比；
    # 工具无 score 阈值过滤（9-RAG.md 遗留 #5），下游无绝对值依赖
    hits = retriever.hybrid_search(query, top_k=top_k)

    return ok(
        {
            "backend": "milvus",
            "query": query,
            "results": [
                {
                    "doc": h.chunk.doc,
                    "section": h.chunk.section,
                    "score": round(h.score, 4),
                    "text": h.chunk.text,
                }
                for h in hits
            ],
        }
    )
