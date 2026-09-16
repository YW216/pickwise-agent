"""初始化 PostgreSQL 真值层：建表（幂等）+ 从本地数据播种。

数据流：mock_data 本地文件（PG 不可达时自动兜底）→ 播种进 PG
       → 此后 mock_data 导入时优先从 PG 加载（真值源切换完成）。

用法：python app/scripts/init_pg.py
前置：deploy/postgres 的 PG 容器已启动
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.db.snapshot import PRODUCTS, FAVORITES  # noqa: E402
from app.db.catalog_repo import init_schema_and_seed, load_full_catalog  # noqa: E402
from app.db.connection import ping  # noqa: E402
from app.config.settings import settings  # noqa: E402


def main():
    url = settings.database_url
    ok, msg = ping(url)
    if not ok:
        print(f"❌ PG 不可达: {msg}")
        print("   请先启动：docker compose -f deploy/postgres/docker-compose.yml up -d")
        sys.exit(1)
    print(f"✅ PG 连通: {url}")

    counts = init_schema_and_seed(
        url, {"products": PRODUCTS, "favorites": FAVORITES}
    )
    if counts.get("skipped"):
        print(f"⏭️ 表已存在且非空（{counts['products']} 款），跳过播种")
    else:
        print(f"✅ 播种完成：商品 {counts['products']} / 收藏 {counts['favorites']}")

    catalog = load_full_catalog(url)
    print(f"回读校验：商品 {len(catalog['products'])} / 收藏 {len(catalog['favorites'])}")
    sample = catalog["products"].get("PH-09")
    if sample:
        print(f"抽查 PH-09: {sample['name']} {sample['price']} 元")


if __name__ == "__main__":
    main()
