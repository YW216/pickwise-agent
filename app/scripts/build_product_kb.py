"""构建商品语义检索库（product_kb collection）。

数据源：app/agent/tools/mock_data.py 的 PRODUCTS（含 introduction 段落）。
每款商品渲染成一张"商品卡片"语义文本，一商品一条记录（chunk_id = product_id）。

卡片模板（text，embedding + BM25 的唯一索引对象，各段均有语义职责）：
  {name}（{category} · {brand}）      ← 型号/品牌词面（BM25 精确匹配型号 query）
  配置：k v / k v …                   ← 参数型 query
  <introduction 段落>                 ← 场景描述最丰富的语义来源
  售价 {price} 元
（summary 不进 text：与 introduction 语义重叠，仅作返回卡片的展示字段）

字段设计（检索面最小集，真值留 catalog / 未来 PG）：
- product_id（主键）→ 命中后回查 catalog 组装返回卡片
- price / category → 标量过滤（预算、品类）
- embedding + sparse_bm25（Function 生成）→ dense / BM25 双路索引

用法：python app/scripts/build_product_kb.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pymilvus import DataType, Function, FunctionType, MilvusClient  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.milvus_utils import ensure_reachable  # noqa: E402
from app.db.snapshot import PRODUCTS  # noqa: E402


def build_card_text(product: dict) -> str:
    """商品 → 卡片语义文本（各段均有语义职责，见模块 docstring）。"""
    lines = [
        f"{product['name']}（{product['category']} · {product['brand']}）",
        "配置：" + " / ".join(f"{k} {v}" for k, v in product["specs"].items()),
        *product.get("introduction", []),
        f"售价 {product['price']} 元",
    ]
    return "\n".join(lines)


def main():
    uri = settings.milvus_uri
    if not uri.startswith("http"):
        uri = str(ROOT / uri)
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
    embedder = Embedder(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
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
    client.create_collection(
        collection_name=collection, schema=schema, index_params=index_params
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

    print("\n🎉 商品语义检索库构建完成。")


if __name__ == "__main__":
    main()
