"""search_products：商品语义搜索（介绍语料，Milvus hybrid 双路）。

与 search_catalog 的分工（Agent 可只调一个，也可都调）：
- search_catalog：结构化硬条件（品牌 / 价格区间 / 参数关键词）精确过滤
- search_products：自然语言模糊需求（「适合打游戏的轻薄本」「通勤安静」），
  按商品介绍卡片的语义（dense）与词面（BM25 型号精确匹配）双路召回

存储：product_kb collection（build_product_kb.py 构建），一商品一条记录，
chunk_id = product_id，text = 商品卡片。Milvus 只存检索面；命中后按
product_id 批量回查 PG 真值组装轻卡片（不依赖进程内快照，价格永远最新）。

硬条件下推（P2-3，2026-10-12）：品类/价格不再用 Milvus 标量表达式过滤
（旧索引的过期标量会漏召回——改价未同步时按旧价放行或误杀），改为先按
PG 当前真值算出符合条件的 ID 集合，再下推 `product_id in [...]` 表达式。
表达式只用仓储返回的 ID 列表构造，不拼接任何用户原文。

评价暂不入库（2026-09-15 起评价与保修数据下线）。
"""

from __future__ import annotations

import json
import threading

from app.agent.rag.embedder import Embedder
from app.agent.rag.milvus_utils import ensure_reachable
from app.agent.tools.catalog import brief_view
from app.agent.tools.result import fail, ok
from app.config.settings import settings
from app.db.catalog_repo import find_eligible_product_ids, get_products_by_ids

_client = None
_embedder = None
_client_lock = threading.Lock()
_embedder_lock = threading.Lock()


def _get_client():
    """懒加载 MilvusClient；http uri 先做 TCP 探活快败（同 MilvusBackend）。

    HTTP 进程中多个窗口的消费者线程会并发调用：模块级缓存的一次初始化
    必须线程安全（4.6），double-checked locking 避免重复建连。
    """
    global _client
    if _client is None:
        with _client_lock:
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
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder(
                    api_key=settings.effective_embedding_api_key,
                    base_url=settings.effective_embedding_base_url,
                    model=settings.effective_embedding_model,
                    timeout=settings.openai_timeout,
                    max_retries=settings.openai_max_retries,
                )
    return _embedder


def close_public_clients() -> None:
    """HTTP 应用退出时显式关闭公共检索客户端（4.6）。

    公共客户端是共享资源，不随某个用户窗口回收而关闭；此入口由
    create_app 的 lifespan 在 SessionManager.close 之后调用。
    """
    global _client, _embedder
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:  # noqa: BLE001 —— 退出清理尽力而为
                pass
        _client = None
    with _embedder_lock:
        _embedder = None


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
    min_price: int | None = None,
    max_price: int | None = None,
    limit: int = 5,
    database_url: str | None = None,
) -> dict:
    """自然语言语义搜索商品：介绍卡片的 dense + BM25 双路召回，RRF 融合。

    Args:
        query：自然语言描述的需求（如「适合打游戏的轻薄本」「通勤戴的安静耳机」，
               也可直接给型号如「凌霄 Pro 14」——BM25 路负责型号精确匹配）
        category：可选品类过滤（笔记本 / 手机 / 耳机）
        min_price：可选价格下限（标量过滤；与 max_price 同值即精确价格）
        max_price：可选预算上限（标量过滤，比"五千左右"这类文本语义可靠）
        limit：返回商品数上限
        database_url：仅程序侧可传的数据库地址（模型 schema 不含此参数）。
            显式传入优先；缺省回退应用配置——服务于「直接函数调用」路径
            （CLI 脚本 / 评测），ToolManager 路径由 bind_tool_handlers
            焊死实例地址。Milvus 检索配置（uri / collection）仍是公共
            资源，不走实例绑定。

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

        url = database_url or settings.database_url
        if not url:
            return fail("商品服务暂未配置，请稍后再试",
                        {"results": [], "total": 0, "unavailable": True})

        # 硬条件（品类/价格）按 PG 当前真值先算 ID 集合，再下推给 Milvus 做
        # `product_id in [...]` 过滤——旧索引的过期标量（改价/改品类尚未同步）
        # 不再漏召回；空集合直接短路为无匹配，不会误当「取消过滤」。
        expr = None
        if category is not None or min_price is not None or max_price is not None:
            eligible = find_eligible_product_ids(
                url, category=category, min_price=min_price, max_price=max_price,
            )
            if not eligible:
                return fail(
                    "没有符合品类/预算条件的商品",
                    {"results": [], "total": 0, "query": query},
                )
            # 表达式只由仓储返回的 ID 列表构成（json 序列化自带转义），
            # 不拼接用户原文——注入面在源头消除
            expr = f"product_id in {json.dumps(eligible, ensure_ascii=False)}"

        # 召回窗口远大于输出量：复核过滤掉个别命中后，槽位由窗口内后续命中
        # 按序补足；窗口取尽仍不足就如实返回较少结果。
        recall = max(limit * 4, 12)
        vector = _get_embedder().encode_one(query)
        ids = _hybrid_product_ids(
            query, vector, top_k=recall, recall_k=recall, expr=expr,
        )

        # 回查 PG 真值组装轻卡片（与 search_catalog 的 brief_view 同构）。
        # 不读进程内快照 PRODUCTS：快照仅在进程启动时加载，PG 改价后即过期；
        # 按命中 id 批量直查（一次 SQL 只取命中几条），价格/名称永远是真值。
        products_by_id = get_products_by_ids(url, ids)
        results = []
        for pid in ids:  # 按 Milvus 检索排序输出，不跟 dict 迭代序
            product = products_by_id.get(pid)
            if not product:
                continue  # 索引尚有、PG 已删/下架：下轮同步清出，先不露出
            # 防御性复核：ID 集合来自 PG 当前值，正常必然命中；保留以防
            # 检索面与真值之间出现视图不一致（如两库之间的并发写）
            if category and product["category"] != category:
                continue
            if min_price is not None and product["price"] < min_price:
                continue
            if max_price is not None and product["price"] > max_price:
                continue
            results.append(brief_view(product))
            if len(results) >= limit:
                break
    except ConnectionError as exc:
        return fail(str(exc), {"results": [], "total": 0, "unavailable": True})
    except Exception as exc:  # noqa: BLE001
        return fail(f"商品语义检索暂时不可用: {exc}",
                    {"results": [], "total": 0, "unavailable": True})

    if not results:
        return fail(
            "没有找到语义相关的商品",
            {"results": [], "total": 0, "query": query},
        )

    return ok({"results": results, "total": len(results), "query": query})
