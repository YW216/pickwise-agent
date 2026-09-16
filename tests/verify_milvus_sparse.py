"""Gate 1：Milvus Lite 稀疏向量机制验证（手造向量，零外部依赖）。

验证：SPARSE_FLOAT_VECTOR 字段 / SPARSE_INVERTED_INDEX 索引 / IP 度量
在 milvus-lite 上能否建、插、搜。
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pymilvus import DataType, MilvusClient

DB_PATH = ROOT / "app/sessions/milvus_sparse_test.db"
COLLECTION = "sparse_gate1"

if DB_PATH.is_dir():
    shutil.rmtree(DB_PATH)
elif DB_PATH.exists():
    DB_PATH.unlink()

client = MilvusClient(str(DB_PATH))

# 建表：一个稀疏字段（Milvus 稀疏向量只支持 IP 度量）
schema = client.create_schema(auto_id=False)
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("text", DataType.VARCHAR, max_length=4096)
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="sparse",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",
)

try:
    client.create_collection(collection_name=COLLECTION, schema=schema, index_params=index_params)
    print("[gate1] 建含 SPARSE_FLOAT_VECTOR 字段的 collection ✅")
except Exception as e:
    print(f"[gate1] 建表失败 ❌ → milvus-lite 不支持稀疏向量：{e}")
    sys.exit(1)

# 插入：词表位 → 权重 的稀疏表示（模拟 3 条"文档"）
docs = {
    "d1": {"text": "文档A：讲 OLED 屏幕的烧屏与对比度", "sparse": {101: 0.9, 205: 0.6, 330: 0.3}},
    "d2": {"text": "文档B：讲 IPS 屏幕的色彩与视角", "sparse": {102: 0.9, 205: 0.5, 411: 0.4}},
    "d3": {"text": "文档C：讲 CPU 处理器性能", "sparse": {501: 0.9, 602: 0.7}},
}
client.insert(collection_name=COLLECTION, data=[
    {"id": k, "text": v["text"], "sparse": v["sparse"]} for k, v in docs.items()
])
print("[gate1] 插入 3 条稀疏向量 ✅")

# 检索：查询向量含 205（两篇屏幕文档都有）+ 101（只有文档A有）→ 期望 top1 = d1
q = {205: 0.5, 101: 0.8}
res = client.search(
    collection_name=COLLECTION,
    data=[q],
    anns_field="sparse",
    limit=3,
    output_fields=["text"],
)[0]
print("[gate1] 稀疏检索结果（期望 top1=d1，d3 相关性≈0）：")
for h in res:
    print(f"   {h['id']} | score {h['distance']:.4f} | {h['entity']['text']}")

top1_ok = res[0]["id"] == "d1"
print(f"\n[gate1 结论] {'✅ milvus-lite 稀疏向量机制可用' if top1_ok else '❌ 排序异常，需排查'}")
