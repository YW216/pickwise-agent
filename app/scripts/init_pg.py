"""初始化 PostgreSQL 真值层：建表（幂等）+ 从本地seed 播种。

数据流：app/db/seed/products.json（冷启动/灾备的唯一播种源）
       → init_schema_and_seed 播种进 PG
       → 此后运行时一律从 PG 读（app/db/snapshot.py 进程内快照）

为什么不由snapshot 播种：snapshot 是 PG 的**消费者**（进程启动时 SELECT
全表），若播种数据也从它取，就是「从 PG 读出来再写回 PG」的自我循环——空库
时_load() 直接抛 RuntimeError，播种 0 条，冷启动死锁。播种源必须是 PG 之外
的独立文件，这也是 init_pg 存在的意义。

JSON 是 mock_data 时代数据文件的搬迁（2026-09-15 随mock_data 废除一起落到
app/db/seed/），非PG dump。**PG 中的数据演进不回写此文件**，故它可能落后于
PG；PG 不可用时用本文件重建即可，事后仍以 PG 为准。

用法：python app/scripts/init_pg.py
前置：deploy/postgres 的 PG 容器已启动
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.db.catalog_repo import init_schema_and_seed, load_full_catalog  # noqa: E402
from app.db.connection import ping  # noqa: E402
from app.config.settings import settings  # noqa: E402

SEED_PATH = ROOT / "app" / "db" / "seed" / "products.json"


def load_seed() -> dict:
    """读播种源。结构与 init_schema_and_seed 的入参一一对应。"""
    if not SEED_PATH.exists():
        raise FileNotFoundError(f"播种源不存在：{SEED_PATH}")
    data = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    for key in ("products", "favorites"):
        if key not in data:
            raise ValueError(f"播种源缺少 {key} 段：{SEED_PATH}")
    if not data["products"]:
        raise ValueError(f"播种源 products 为空：{SEED_PATH}")
    return data


def main():
    url = settings.database_url
    ok, msg = ping(url)
    if not ok:
        print(f"❌ PG 不可达: {msg}")
        print("   请先启动：docker compose -f deploy/postgres/docker-compose.yml up -d")
        sys.exit(1)
    print(f"✅ PG 连通: {url}")

    seed = load_seed()
    meta = seed.get("meta", {})
    print(f"📄 播种源：{SEED_PATH.relative_to(ROOT)}"
          f"（导出于 {meta.get('exported', '未知')}）")
    print(f"   待播种：商品 {len(seed['products'])} / 收藏 {len(seed['favorites'])}")

    counts = init_schema_and_seed(
        url, {"products": seed["products"], "favorites": seed["favorites"]}
    )
    # init_schema_and_seed 只在表为空时播种，「是否跳过」看它回读的 existing_products
    if counts["existing_products"]:
        print(f"⏭️ 表已存在且非空（{counts['existing_products']} 款），"
              "跳过播种——本文件不覆盖已有数据")
    else:
        print(f"✅ 播种完成：商品 {counts['products']} / 收藏 {counts['favorites']}")

    # 回读校验：条数对≠ 内容对（2026-09-14 错位事故的教训）
    catalog = load_full_catalog(url)
    print(f"回读校验：商品 {len(catalog['products'])} / 收藏 {len(catalog['favorites'])}")
    sample = catalog["products"].get("PH-09")
    if sample:
        print(f"抽查 PH-09: {sample['name']} {sample['price']} 元")

    # 播种源与 PG 的差异提示：非零说明 PG 已独立演进（或 seed 已过期），
    # 不阻断——PG 是真值源，这里只让人知道两边不再一致
    n_pg, n_seed = len(catalog["products"]), len(seed["products"])
    if n_pg != n_seed:
        print(f"⚠️ 条数不一致：PG {n_pg} vs 播种源 {n_seed}。"
              "以 PG 为准；如需重建请先清空 products/favorites 表再跑本脚本。")
    if len(catalog["favorites"]) != len(seed["favorites"]):
        print(f"⚠️ 收藏数不一致：PG {len(catalog['favorites'])} vs "
              f"播种源 {len(seed['favorites'])}")


if __name__ == "__main__":
    main()