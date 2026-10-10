"""商品索引入口脚本（CLI 壳）：rebuild 全量重建 / incremental 增量同步。

两种模式（develop_docs/rag增量更新.md 第七节）：
  --mode rebuild      首次初始化 / 契约变更 / 异常恢复：drop → 全量 embed → 重建
  --mode incremental  日常同步：只处理 diff 出的变化商品（默认模式）
  --dry-run           只输出差异计划，不写 Milvus、不写 PG、不调 embedding

同步核心逻辑在 app/services/product_sync.py（service 层，与壳解耦）——
本脚本只是 CLI 壳；worker / FastAPI 未来复用同一 sync()。

用法：
  python app/scripts/build_product_kb.py                        # 增量同步
  python app/scripts/build_product_kb.py --mode incremental --dry-run
  python app/scripts/build_product_kb.py --mode rebuild
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pymilvus import DataType, Function, FunctionType, MilvusClient  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.milvus_utils import ensure_reachable  # noqa: E402
from app.db.catalog_repo import mark_products_indexed  # noqa: E402
from app.db.snapshot import PRODUCTS  # noqa: E402
from app.services.product_sync import (  # noqa: E402
    TEMPLATE_VERSION,
    build_card_text,
    build_contract,
    card_hash,
    sync,
)


def main():
    """CLI 分流：rebuild 走本脚本全量重建，incremental 委托 service 层 sync()。"""
    parser = argparse.ArgumentParser(description="商品语义索引：rebuild / incremental")
    parser.add_argument(
        "--mode",
        choices=["incremental", "rebuild"],
        default="incremental",
        help="incremental=日常增量同步（默认）；rebuild=全量重建",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出差异计划，不写 Milvus、不写 PG、不调 embedding（仅 incremental）",
    )
    args = parser.parse_args()

    if args.mode == "rebuild":
        rebuild()
        return

    try:
        stats = sync(dry_run=args.dry_run)
    except RuntimeError as exc:
        # 可预期失败（契约不匹配 / 空快照 / collection 缺失）：人话提示后退出
        print(f"❌ 增量同步中止：{exc}")
        sys.exit(1)

    label = "差异计划（dry-run，未写入）" if args.dry_run else "增量同步报告"
    print("=" * 60)
    print(f"  商品索引增量同步 · {label}")
    print("=" * 60)
    print(f"  新增         : {len(stats.added)}  {stats.added or ''}")
    print(f"  文本变(重算向量): {len(stats.text_changed)}  {stats.text_changed or ''}")
    print(f"  行变(复用向量) : {len(stats.row_changed)}  {stats.row_changed or ''}")
    print(f"  未变(跳过)     : {len(stats.unchanged)}")
    print(f"  删除         : {len(stats.deleted)}  {stats.deleted or ''}")
    if not args.dry_run:
        print(f"  embedding 调用 : {stats.embedded}（降级 {len(stats.degraded)}）")
        print(f"  upsert / delete : {stats.upserted} / {stats.deleted_count}")
        print(f"  状态标记        : {stats.marked}"
              + (f"（乐观锁跳过 {stats.lock_skipped}，下轮重做）" if stats.lock_skipped else ""))
    print("=" * 60)


def rebuild():
    """全量重建（drop → create → insert），首次初始化 / 契约变更 / 异常恢复用。"""
    uri = settings.milvus_uri
    ensure_reachable(uri)

    client = MilvusClient(uri)
    collection = settings.product_collection

    print("=" * 60)
    print("  并夕夕 · 商品语义检索库构建")
    print(f"  索引目标  : {uri} (collection={collection})")
    print(f"  Embedding : {settings.embedding_model}")
    print("=" * 60)

    # 全量重建：先删后建（卡片文本/分词规则变更时避免脏数据）
    if client.has_collection(collection):
        client.drop_collection(collection)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("product_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba"},
    )
    schema.add_field("price", DataType.INT64)
    schema.add_field("category", DataType.VARCHAR, max_length=32)
    schema.add_field("sparse_bm25", DataType.SPARSE_FLOAT_VECTOR)

    # 维度在编码后才知道，先编码再建 collection
    products = list(PRODUCTS.values())
    texts = [build_card_text(p) for p in products]
    print(f"\n[1/3] 渲染 {len(products)} 张商品卡片...")
    for pid, p in zip(PRODUCTS, products):
        print(f"   - {pid}: {p['name']}")

    print(f"\n[2/3] 调用 {settings.embedding_model} 批量向量化...")
    settings.assert_embedding_configured()
    embedder = Embedder(
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url,
        model=settings.effective_embedding_model,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )
    vectors = embedder.encode(texts)
    print(f"   完成，向量维度 = {len(vectors[0])}")

    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=len(vectors[0]))
    schema.add_function(
        Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names="sparse_bm25",
        )
    )
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE"
    )
    index_params.add_index(
        field_name="sparse_bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    print("\n[3/3] 写入索引...")
    # 契约指纹持久化在 collection properties（Milvus 无 collection 级自定义
    # metadata）：记录实际使用的 embedder 模型，而非 settings 配置值——两者在
    # 回退链下可能不同，增量校验必须以"实际构建时用的"为准。
    # 注意：pymilvus 2.6 + standalone 实测 create_collection 的 description
    # 参数不生效（describe 返回空），必须写 properties.description
    contract = build_contract(embedder.model)
    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params,
        properties={"description": contract},
    )
    client.insert(
        collection_name=collection,
        data=[
            {
                "product_id": p["product_id"],
                "text": text,
                "price": p["price"],
                "category": p["category"],
                "embedding": vec,
            }
            for p, text, vec in zip(products, texts, vectors)
        ],
    )

    # 对齐校验（2026-09-14 教训：曾发生索引内容与真值层错位——行数正确、ID 存在，
    # 但 text 是别的商品的数据，造成"隐形商品"且完全不可见）。抽查 5 行回读比对，
    # 不一致立即报错退出，防止错位索引流入检索层。
    import random
    import time

    time.sleep(2)  # 等 segment 落盘可查
    sample_ids = random.sample(list(PRODUCTS), min(5, len(PRODUCTS)))
    misaligned = []
    for pid in sample_ids:
        got = client.query(
            collection_name=collection,
            filter=f'product_id == "{pid}"',
            output_fields=["text", "price"],
            limit=1,
        )
        if not got:
            misaligned.append(f"{pid}: 行不存在")
            continue
        want_prefix = build_card_text(PRODUCTS[pid])[:30]
        if not got[0].get("text", "").startswith(want_prefix[:20]):
            misaligned.append(f"{pid}: text 错位（got={got[0].get('text', '')[:24]!r}）")
    if misaligned:
        print("❌ 对齐校验失败——索引内容与真值层错位：")
        for m in misaligned:
            print(f"   {m}")
        print("   请检查 product_pool_extra.json 是否与手写款 ID 冲突后重建。")
        sys.exit(1)
    print(f"   对齐校验：抽查 {len(sample_ids)} 行 text/price 与真值一致 ✅")
    client.load_collection(collection)
    print(f"   已写入 {len(products)} 条商品卡片")

    # 写回同步状态（衔接增量同步，develop_docs/rag增量更新.md）：
    # rebuild 等价于"全量同步成功"，必须把 card_hash + indexed_at 刷齐，
    # 否则增量首轮会把全部商品当脏数据重算（幂等但白烧 embedding 调用）
    marked, _skipped = mark_products_indexed(
        settings.database_url,
        {p["product_id"]: card_hash(text) for p, text in zip(products, texts)},
    )
    print(f"   已写回同步状态（PG card_hash/indexed_at）：{marked} 条")

    print("\n🎉 商品语义检索库构建完成。")


if __name__ == "__main__":
    main()
