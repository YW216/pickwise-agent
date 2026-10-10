"""知识库检索器：query → 向量化 → 委托 MilvusBackend 检索。

设计上 retriever 只负责"问句怎么变向量""结果怎么聚合"，
存储和打分都交给 MilvusBackend——standalone 部署（deploy/milvus）一套 API。

校验逻辑：加载后比对 backend 持久化的 embedding_model 与当前 Embedder.model，
不一致直接报错——避免"换了 embedding 但还在用老索引"这种隐蔽问题。
"""

from __future__ import annotations

from app.agent.rag.embedder import Embedder
from app.agent.rag.milvus_backend import MilvusBackend, RetrievedChunk

__all__ = ["KnowledgeRetriever", "RetrievedChunk"]


class KnowledgeRetriever:
    """对上层暴露统一接口，对下委托给 MilvusBackend。"""

    def __init__(self, embedder: Embedder, backend: MilvusBackend):
        self._embedder = embedder
        self._backend = backend
        self._loaded = False

    @property
    def backend(self) -> MilvusBackend:
        return self._backend

    @property
    def size(self) -> int:
        return self._backend.size()

    def load(self) -> None:
        if self._loaded:
            return
        self._backend.load()

        expected = self._backend.expected_embedding_model()
        if expected and expected != self._embedder.model:
            raise ValueError(
                f"索引模型({expected}) 与当前 Embedder 模型"
                f"({self._embedder.model}) 不一致，请重建索引。"
            )
        self._loaded = True

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not self._loaded:
            self.load()
        q_vec = self._embedder.encode_one(query)
        return self._backend.search(q_vec, top_k=top_k)

    def hybrid_search(
        self, query: str, top_k: int = 5, recall_k: int | None = None
    ) -> list[RetrievedChunk]:
        """混合检索：dense 语义 + BM25 关键词，服务端 RRF 融合。

        query 原文一路直通 BM25（服务端分词打分），另一路由 embedder
        编码成向量走 dense——retriever 是唯一同时持有原文与 embedder
        的层，所以透传方法放这里而不是上层。
        recall_k：每路召回窗口，缺省由 backend 按漏斗原则决定（评测可传小值做窄窗对照）。
        """
        if not self._loaded:
            self.load()
        q_vec = self._embedder.encode_one(query)
        return self._backend.hybrid_search(query, q_vec, top_k=top_k, recall_k=recall_k)
