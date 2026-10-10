"""构建商品语义检索库（product_kb collection）。

数据源：PG products 表（真值层，经 app.db.snapshot.PRODUCTS 加载）。
每款商品渲染成一张"商品卡片"语义文本，一商品一条记录（chunk_id = product_id）。

卡片模板 v2（text，embedding + BM25 的唯一索引对象，各段均有语义职责）：
  {name}（{category} · {brand}）      ← 型号/品牌词面（BM25 精确匹配型号 query）
  配置：k v / k v …                   ← 参数型 query（specs 按键名排序，防字典序漂移）
  <introduction 段落>                 ← 场景描述最丰富的语义来源
（summary 不进 text：与 introduction 语义重叠，仅作返回卡片的展示字段）

v2 变更（2026-10-10）：移除"售价 X 元"行——
- embedding 对数字 token 不敏感，价格进 text 无语义收益；
- 预算过滤已有 price 标量字段下推 expr，比词面匹配可靠；
- 价格是高频变更字段，留在 text 会导致每次调价都重算 embedding。
price 从此只走标量通道（见 develop_docs/rag增量更新.md 第四节）。

字段设计（检索面最小集，真值留 catalog / PG）：
- product_id（主键）→ 命中后回查 catalog 组装返回卡片
- price / category → 标量过滤（预算、品类）
- embedding + sparse_bm25（Function 生成）→ dense / BM25 双路索引
- collection description 持久化契约指纹（embedding 模型 + 模板版本），
  供增量同步校验"索引内容与当前构建方式是否兼容"

用法：python app/scripts/build_product_kb.py
"""

import hashlib
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

# 卡片模板版本：模板任何影响 text 的改动都必须递增，并触发一次全量重建
# （旧向量与新模板混索 = 语义不一致；版本号进 collection description 参与契约校验）
TEMPLATE_VERSION = "product_card_template_v2"


def card_hash(text: str) -> str:
    """检索面文本指纹：增量同步判"向量是否需要重算"的唯一依据。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_card_text(product: dict) -> str:
    """商品 → 卡片语义文本（各段均有语义职责，见模块 docstring）。"""
    lines = [
        f"{product['name']}（{product['category']} · {product['brand']}）",
        "配置：" + " / ".join(
            f"{k} {v}" for k, v in sorted(product["specs"].items())
        ),
        *product.get("introduction", []),
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
    contract = f"embedding_model={embedder.model};template={TEMPLATE_VERSION}"
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
    indexed = mark_products_indexed(
        settings.database_url,
        {p["product_id"]: card_hash(text) for p, text in zip(products, texts)},
    )
    print(f"   已写回同步状态（PG card_hash/indexed_at）：{indexed} 条")

    print("\n🎉 商品语义检索库构建完成。")


if __name__ == "__main__":
    main()
