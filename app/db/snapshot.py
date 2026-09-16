"""真值层进程内快照：进程启动时从 PG 加载一次，全项目共享（2026-09-15）。

**PG 是唯一数据源**（2026-09-15 裁定：mock_data 本地数据副本删除——
两份数据并存必然漂移，索引错位事故的根源）。加载失败 = 目录服务不可用，
消费方（工具）显式失败，不再静默回退旧快照。

快照语义：进程生命周期内数据不变；PG 数据变更需重启进程（未来 FastAPI
服务化时改为 repository 直查 / TTL 缓存，接口不变）。

seed 文件（app/db/seed/products.json）是 PG 的导出快照，仅供 init_pg
重建/灾备播种——不是运行时数据源。
"""

from app.db.catalog_repo import load_full_catalog
from app.config.settings import settings

# 真值层快照（进程启动时由 _load() 填充）
PRODUCTS: dict[str, dict] = {}
FAVORITES: list[str] = []


def _load() -> None:
    if not settings.database_url:
        raise RuntimeError("未配置 database_url，商品目录服务不可用")
    catalog = load_full_catalog(settings.database_url)
    if not catalog["products"]:
        raise RuntimeError("PG 商品表为空（未播种），请先运行 app/scripts/init_pg.py")
    PRODUCTS.update(catalog["products"])
    FAVORITES.extend(catalog["favorites"])


try:
    _load()
    print(f"[snapshot] 真值层已从 PostgreSQL 加载：商品 {len(PRODUCTS)} / 收藏 {len(FAVORITES)}")
except Exception as exc:  # noqa: BLE001 —— 快照为空，工具层显式失败（不静默回退旧数据）
    print(f"[snapshot] ⚠️ 真值层加载失败，商品目录服务不可用: {type(exc).__name__}: {exc}")
