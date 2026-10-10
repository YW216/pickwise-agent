"""Milvus 向量后端：基于 MilvusClient 接入 standalone（HTTP 服务）。

- 部署：deploy/milvus 下 docker compose（standalone + Attu），uri 走 settings.milvus_uri
- 稀疏向量 + BM25 Function（服务端 jieba 分词）支撑混合检索（dense + BM25 + RRF）

schema 约定（探路阶段实测确认）：
- 字段：chunk_id（主键）/ doc / section / text / embedding
- 索引：AUTOINDEX + COSINE（score = 相似度，越大越相似）
- embedding_model 借道 collection properties 持久化（pymilvus 2.6 + standalone
  实测 description 参数不生效）：加载时校验，避免"换了 embedding 却还在用老索引"
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.rag.chunker import Chunk
from app.agent.rag.milvus_utils import ensure_reachable

__all__ = ["MilvusBackend", "RetrievedChunk"]


@dataclass
class RetrievedChunk:
    """检索结果：命中 chunk + 相似度得分（COSINE，越大越相似）。"""

    chunk: Chunk
    score: float


class MilvusBackend:
    """基于 MilvusClient 的向量后端：dense 单路（search）+ 混合检索（hybrid_search）。"""

    def __init__(self, uri: str, collection_name: str = "ecom_kb"):
        self._uri = uri
        self._collection_name = collection_name
        self._client = None
        self._embedding_model: str = ""
        self._loaded = False
        self._size_cache: int | None = None

    # ---- 内部工具 ----

    def _ensure_client(self) -> None:
        """懒加载 MilvusClient；http uri 先做 TCP 探活快败（详见 milvus_utils）。"""
        if self._client is not None:
            return
        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise ImportError(
                "pymilvus 未安装。请运行 `pip install pymilvus` 或 "
                "`pip install -r requirements.txt`。"
            ) from e

        ensure_reachable(self._uri)
        self._client = MilvusClient(self._uri)

    def _require_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ---- VectorBackend 契约（构建 / 检索 / 元信息） ----

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        embedding_model: str,
    ) -> None:
        """全量重建索引（覆盖式）：drop → create → insert。"""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks 与 vectors 长度不一致: {len(chunks)} vs {len(vectors)}"
            )
        if not chunks:
            raise ValueError("chunks 为空，拒绝构建空索引")

        self._ensure_client()

        # 全量重建：先删后建，避免 embedding 模型切换时的脏数据
        if self._client.has_collection(self._collection_name):
            self._client.drop_collection(self._collection_name)

        from pymilvus import DataType, Function, FunctionType

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("doc", DataType.VARCHAR, max_length=256)
        schema.add_field("section", DataType.VARCHAR, max_length=512)
        # jieba 分词器：BM25 Function 在服务端用它对 text 分词、生成稀疏向量
        # （参数格式为 {"tokenizer": "jieba"}，服务端文档的
        #   {"type": "chinese"} 写法不被接受）
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "jieba"},
        )
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=len(vectors[0]))
        schema.add_field("sparse_bm25", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names="sparse_bm25",
            )
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_bm25",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        # 契约指纹写 collection properties（pymilvus 2.6 + standalone 实测
        # description 参数不生效，properties.description 才能被 describe 读回）
        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
            properties={"description": embedding_model},
        )

        self._client.insert(
            collection_name=self._collection_name,
            data=[
                {
                    "chunk_id": c.chunk_id,
                    "doc": c.doc,
                    "section": c.section,
                    "text": c.text,
                    "embedding": vec,
                }
                for c, vec in zip(chunks, vectors)
            ],
        )

        self._embedding_model = embedding_model
        self._size_cache = len(chunks)
        self._loaded = True

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """dense 检索：COSINE 相似度 Top-K（score 越大越相似）。"""
        self._require_loaded()

        hits = self._client.search(
            collection_name=self._collection_name,
            data=[query_vector],
            anns_field="embedding",
            limit=top_k,
            # pymilvus 2.x：主键不会默认带回，必须显式声明才会出现在 entity 里
            output_fields=["chunk_id", "doc", "section", "text"],
        )[0]

        out: list[RetrievedChunk] = []
        for h in hits:
            entity = h["entity"]
            chunk = Chunk(
                chunk_id=entity["chunk_id"],
                doc=entity["doc"],
                section=entity["section"],
                text=entity["text"],
            )
            out.append(RetrievedChunk(chunk=chunk, score=float(h["distance"])))
        return out

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        recall_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """混合检索：dense 语义路 + BM25 精确词路，服务端 RRF 按排名融合。

        - dense 路：接收 query 向量（调用方编码，retriever 手里同时有原文与 embedder）
        - BM25 路：直接接收原始文本，服务端用建库时的 jieba analyzer 分词打分
        - 融合分数是 RRF 值（约 1/(60+rank) 量级），与 dense 的 COSINE 分数
          量纲不同，两者不可直接比较，只能各自内部排序
        - recall_k：每路召回窗口（漏斗原则：远大于 top_k，给双路共识留检测窗口）；
          缺省 max(top_k*5, 20)，评测脚本可传小值做窄窗对照
        """
        self._require_loaded()
        from pymilvus import AnnSearchRequest, RRFRanker

        # 漏斗原则：每路召回量远大于最终输出量。若每路只取 top_k，
        # "另一路其实也认可"的 chunk 会被召回窗口切掉，拿不到双路共识加成
        rk = recall_k or max(top_k * 5, 20)

        req_dense = AnnSearchRequest(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=rk,
        )
        req_bm25 = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_bm25",
            param={"metric_type": "BM25"},
            limit=rk,
        )

        hits = self._client.hybrid_search(
            collection_name=self._collection_name,
            reqs=[req_dense, req_bm25],
            ranker=RRFRanker(),
            limit=top_k,
            output_fields=["chunk_id", "doc", "section", "text"],
        )[0]

        out: list[RetrievedChunk] = []
        for h in hits:
            entity = h["entity"]
            chunk = Chunk(
                chunk_id=entity.get("chunk_id", h.get("id", "")),
                doc=entity["doc"],
                section=entity["section"],
                text=entity["text"],
            )
            out.append(RetrievedChunk(chunk=chunk, score=float(h["distance"])))
        return out

    def size(self) -> int:
        """当前已索引的 chunk 数量（get_collection_stats 懒查询 + 缓存）。"""
        self._require_loaded()
        if self._size_cache is None:
            stats = self._client.get_collection_stats(self._collection_name)
            self._size_cache = int(stats.get("row_count", 0))
        return self._size_cache

    def load(self) -> None:
        """连接并校验 collection 存在；不存在时抛 FileNotFoundError。"""
        self._ensure_client()
        if not self._client.has_collection(self._collection_name):
            raise FileNotFoundError(
                f"Milvus collection '{self._collection_name}' 不存在于 "
                f"{self._uri}。请先运行 `python app/scripts/build_kb_index.py` "
                f"构建索引。"
            )
        info = self._client.describe_collection(self._collection_name)
        # 优先读 properties.description（standalone 实测唯一生效的写法）
        props = info.get("properties") or {}
        self._embedding_model = (
            props.get("description") or info.get("description") or ""
        ).strip()
        # 跨进程重开持久化库时 collection 处于 released 状态，搜索前必须 load
        # （同进程 create→insert 会自动 load，因此重建场景下此调用是无害的幂等操作）
        self._client.load_collection(self._collection_name)
        self._size_cache = None
        self._loaded = True

    def expected_embedding_model(self) -> str:
        """已持久化索引使用的 embedding 模型名（供 retriever 校验）。"""
        self._require_loaded()
        return self._embedding_model
