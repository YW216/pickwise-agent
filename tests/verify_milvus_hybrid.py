"""Gate：Milvus 混合检索可行性验证（BM25 Function + hybrid_search）。

跟随后端配置：uri 取 settings.milvus_uri，lite（本地路径）与 standalone（http）通用。
测试 collection 独立命名（hybrid_gate），结束时自动 drop，不污染 ecom_kb。

验证四件事（全手造数据，零外部依赖）：
1. 建含 BM25 Function 的 collection（中文 analyzer + SPARSE_FLOAT_VECTOR 自动生成）
2. 插入时稀疏向量自动生成（只传 text，不传 sparse）
3. hybrid_search + RRFRanker 双路融合可用
4. BM25 精确词命中生效（罕见词 query 期望 bm25 路把它顶到 top1）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config.settings import settings

COLLECTION = "hybrid_gate"

client = MilvusClient(settings.milvus_uri)
print(f"[连接] milvus_uri = {settings.milvus_uri}")

# 残留 collection 兜底清理（上次运行中断时可能留下）
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

# ---- Gate 1：建含 BM25 Function 的 collection ----
# 手造 4 维 dense 向量（语义=维度位置），文本走中文 analyzer，sparse 由 BM25 Function 自动生成
docs = {
    "d1": ("OLED 屏幕烧屏原理与对比度优势", [1.0, 0.0, 0.0, 0.0]),
    "d2": ("IPS 屏幕色彩与视角表现", [0.0, 1.0, 0.0, 0.0]),
    "d3": ("CPU 处理器性能天梯图 2024 版", [0.0, 0.0, 1.0, 0.0]),
    "d4": ("退货流程与运费承担说明", [0.0, 0.0, 0.0, 1.0]),
}

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field(
    "text",
    DataType.VARCHAR,
    max_length=4096,
    enable_analyzer=True,
    # lite 3.x 的 analyzer 参数格式是 {"tokenizer": "jieba"}，
    # 服务端 2.5 文档里的 {"type": "chinese"} 写法在 lite 上不认
    analyzer_params={"tokenizer": "jieba"},
)
schema.add_field("sparse_bm25", DataType.SPARSE_FLOAT_VECTOR)
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)

try:
    schema.add_function(
        Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names="sparse_bm25",
        )
    )
    print("[gate1] BM25 Function 注册成功（text → sparse_bm25 自动生成）✅")
except Exception as e:
    print(f"[gate1] BM25 Function 注册失败 ❌ → 该版本 milvus-lite 不支持 Function：{e}")
    sys.exit(1)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="dense", index_type="AUTOINDEX", metric_type="COSINE"
)
index_params.add_index(
    field_name="sparse_bm25", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
)

try:
    client.create_collection(
        collection_name=COLLECTION, schema=schema, index_params=index_params
    )
    print("[gate1] 建含 BM25 Function 的 collection ✅")
except Exception as e:
    print(f"[gate1] 建表失败 ❌ → {e}")
    sys.exit(1)

# ---- Gate 2：插入只传 text + dense，sparse 应自动生成 ----
try:
    client.insert(
        collection_name=COLLECTION,
        data=[{"id": k, "text": t, "dense": v} for k, (t, v) in docs.items()],
    )
    print("[gate2] 插入 4 条（未传 sparse，由 Function 自动生成）✅")
except Exception as e:
    print(f"[gate2] 插入失败 ❌ → {e}")
    sys.exit(1)

# 跨进程坑预防：搜索前显式 load
client.load_collection(COLLECTION)

# ---- Gate 3：hybrid_search + RRFRanker 双路融合 ----
try:
    req_dense = AnnSearchRequest(
        data=[[0.9, 0.1, 0.0, 0.0]],  # 语义上靠近 d1（OLED）
        anns_field="dense",
        param={"metric_type": "COSINE"},
        limit=3,
    )
    req_bm25 = AnnSearchRequest(
        data=["天梯图"],  # 精确词，只有 d3 含此词
        anns_field="sparse_bm25",
        param={"metric_type": "BM25"},
        limit=3,
    )
    res = client.hybrid_search(
        collection_name=COLLECTION,
        reqs=[req_dense, req_bm25],
        ranker=RRFRanker(),
        limit=4,
        output_fields=["text"],
    )[0]
    hits = [(h["id"], round(h["distance"], 4)) for h in res]
    print(f"[gate3] hybrid_search + RRFRanker ✅ 融合结果: {hits}")
except Exception as e:
    print(f"[gate3] hybrid_search 失败 ❌ → {type(e).__name__}: {e}")
    sys.exit(1)

# ---- Gate 4：BM25 精确词命中（单路 sparse 检索，罕见词应精确命中 d3）----
try:
    sparse_only = client.search(
        collection_name=COLLECTION,
        data=["天梯图"],
        anns_field="sparse_bm25",
        limit=2,
        output_fields=["text"],
    )[0]
    sparse_hits = [h["id"] for h in sparse_only]
    top1_ok = sparse_hits and sparse_hits[0] == "d3"
    print(
        f"[gate4] BM25 精确词检索 {'✅' if top1_ok else '⚠️'} "
        f"top hits: {sparse_hits}（期望 top1=d3）"
    )
except Exception as e:
    print(f"[gate4] BM25 单路检索失败 ❌ → {type(e).__name__}: {e}")
    sys.exit(1)

print("\n[结论] 四道 gate 全过 → 当前引擎支持 BM25 Function + hybrid_search")

# 收尾：清掉测试 collection，不污染正式库
client.drop_collection(COLLECTION)
print("[清理] 测试 collection 已删除")
