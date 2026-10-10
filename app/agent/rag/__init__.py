"""RAG 模块：知识库切分、向量化、检索。

第 5 期：让 Agent 能基于 FAQ、退换货政策、配送说明、会员权益等
非结构化知识回答问题，而不只依赖工具返回的结构化数据。

向量后端收敛为 MilvusBackend 单实现（2026-09-10 裁定），
standalone 部署（deploy/milvus）一套 API 服务开发与生产，抽象层不再有存在收益。
"""

from app.agent.rag.chunker import Chunk, chunk_markdown_dir
from app.agent.rag.embedder import Embedder
from app.agent.rag.milvus_backend import MilvusBackend, RetrievedChunk
from app.agent.rag.retriever import KnowledgeRetriever

__all__ = [
    "Chunk",
    "chunk_markdown_dir",
    "Embedder",
    "KnowledgeRetriever",
    "MilvusBackend",
    "RetrievedChunk",
]
