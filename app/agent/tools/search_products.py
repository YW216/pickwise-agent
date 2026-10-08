"""search_products：商品语义搜索（介绍语料，Milvus hybrid 双路）。

与 search_catalog 的分工（Agent 可只调一个，也可都调）：
- search_catalog：结构化硬条件（品牌 / 价格区间 / 参数关键词）精确过滤
- search_products：自然语言模糊需求（「适合打游戏的轻薄本」「通勤安静」），
  按商品介绍卡片的语义（dense）与词面（BM25 型号精确匹配）双路召回

存储：product_kb collection（build_product_kb.py 构建），一商品一条记录，
chunk_id = product_id，text = 商品卡片。Milvus 只存检索面；返回时按
product_id 回查 catalog 组装轻卡片（真值在 catalog / 未来 PG）。

评价暂不入库（2026-09-15 起评价与保修数据下线）。
"""

from __future__ import annotations

from app.agent.rag.embedder import Embedder
from app.agent.rag.milvus_utils import ensure_reachable
from app.agent.tools.catalog import brief_view
from app.db.snapshot import PRODUCTS
from app.agent.tools.result import fail, ok
from app.config.settings import settings

_client = None
_embedder = None


def _get_client():
    """懒加载 MilvusClient；http uri 先做 TCP 探活快败（同 MilvusBackend）。"""
    global _client
    if _client is None:
        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise ImportError(
                "pymilvus 未安装。请运行 `pip install pymilvus` 或 "
                "`pip install -r requirements.txt`。"
            ) from e
        ensure_reachable(settings.milvus_uri)
        _client = MilvusClient(settings.milvus_uri)
    return _client


def _get_embedder() -> Embedder:
    """复用 P5 知识库的 embedding 配置（与 retrieve_knowledge 同一通路）。"""
    global _embedder
    if _embedder is None:
        _embedder = Embedder(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            timeout=settings.openai_timeout,
            max_retries=settings.openai_max_retries,
        )
    return _embedder


def _build_expr(category: str | None, max_price: int | None) -> str | None:
    """标量过滤表达式，下推到两路召回（过滤在 top-k 截断前生效）。"""
    parts = []
    if category:
        parts.append(f"category == '{category}'")
    if max_price is not None:
        parts.append(f"price <= {int(max_price)}")
    return " and ".join(parts) or None


def _hybrid_product_ids(
    query: str,
    query_vector: list[float],
    top_k: int,
    recall_k: int | None = None,
    expr: str | None = None,
) -> list[str]:
    """商品库 hybrid 检索核心（生产与评测共用，保证评测不分叉）。

    dense 收向量 / BM25 收原文，RRF 融合，返回按排序的 product_id 列表。
    recall_k 缺省 = max(top_k*4, 12)（漏斗原则：召回窗口远大于输出量）。
    调用前须确保 collection 存在且已 load。
    """
    from pymilvus import AnnSearchRequest, RRFRanker

    rk = recall_k or max(top_k * 4, 12)
    client = _get_client()
    req_dense = AnnSearchRequest(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=rk,
        expr=expr,
    )
    req_bm25 = AnnSearchRequest(
        data=[query],
        anns_field="sparse_bm25",
        param={"metric_type": "BM25"},
        limit=rk,
        expr=expr,
    )
    hits = client.hybrid_search(
        collection_name=settings.product_collection,
        reqs=[req_dense, req_bm25],
        ranker=RRFRanker(),
        limit=top_k,
        output_fields=["product_id"],
    )[0]
    return [
        h["entity"].get("product_id", h.get("id", "")) for h in hits
    ]


def search_products(
    query: str,
    category: str | None = None,
    max_price: int | None = None,
    limit: int = 5,
) -> dict:
    """自然语言语义搜索商品：介绍卡片的 dense + BM25 双路召回，RRF 融合。

    Args:
        query：自然语言描述的需求（如「适合打游戏的轻薄本」「通勤戴的安静耳机」，
               也可直接给型号如「凌霄 Pro 14」——BM25 路负责型号精确匹配）
        category：可选品类过滤（笔记本 / 手机 / 耳机）
        max_price：可选预算上限（标量过滤，比"五千左右"这类文本语义可靠）
        limit：返回商品数上限

    Returns:
        成功：data = {"results": [brief_view 卡片 + summary + score], "total": N}
        索引未初始化 / 检索服务不可用 / 无结果：fail（含可操作提示）
    """
    if not query or not query.strip():
        return fail("请描述你想找的商品")

    try:
        client = _get_client()
        collection = settings.product_collection
        if not client.has_collection(collection):
            return fail(
                "商品语义索引未初始化，请先运行 build_product_kb.py",
                {"results": [], "total": 0},
            )
        client.load_collection(collection)

        expr = _build_expr(category, max_price)
        ids = _hybrid_product_ids(
            query,
            _get_embedder().encode_one(query),
            top_k=limit,
            expr=expr,
        )
    except ConnectionError as exc:
        return fail(str(exc), {"results": [], "total": 0})
    except Exception as exc:  # noqa: BLE001 —— 检索服务故障时优雅降级
        return fail(f"商品语义检索暂时不可用: {exc}")

    # 回查 catalog 组装轻卡片（与 search_catalog 的 brief_view 同构）
    results = []
    for pid in ids:
        product = PRODUCTS.get(pid)
        if not product:
            continue
        card = brief_view(product)
        results.append(card)

    if not results:
        return fail(
            "没有找到语义相关的商品",
            {"results": [], "total": 0, "query": query},
        )

    return ok({"results": results, "total": len(results), "query": query})
