"""商品目录仓储：PG 真值层的查询入口（search_catalog 工具与真值加载共用）。

可搜索字段（与 PG products 表列一一对应，全部确定可枚举）：
- name / brand：ILIKE 字面子串（大小写不敏感）
- category：精确匹配
- price：数值上限
**specs 与 introduction 不参与本仓储的过滤**——参数条件与语义需求由
search_products（Milvus hybrid）承担。

brief 形状（与 tools/catalog.brief_view 一致，Agent 合并零成本）：
  {product_id, name, brand, category, price, summary}
summary = introduction 首段截断 64 字。
"""

import json

import psycopg

from app.db.connection import connect


def search_candidates(
    database_url: str,
    query: str | None = None,
    brand: str | None = None,
    category: str | None = None,
    budget_max: float | None = None,
    limit: int = 10,
) -> dict:
    """结构化过滤查询：品牌/型号字面 + 品类精确 + 价格上限。

    Returns:
        {"candidates": [brief...], "total": N}
    """
    sql = """
        SELECT product_id, name, brand, category, price, LEFT(introduction->>0, 64)
        FROM products
        WHERE (%(cat)s::text IS NULL OR category = %(cat)s::text)
          AND (%(brand)s::text IS NULL OR brand = %(brand)s::text)
          AND (%(budget)s::numeric IS NULL OR price <= %(budget)s::numeric)
          AND (%(q)s::text IS NULL OR name ILIKE '%%' || %(q)s::text || '%%'
                                 OR brand ILIKE '%%' || %(q)s::text || '%%')
        ORDER BY product_id
        LIMIT %(lim)s
    """
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"cat": category, "brand": brand, "budget": budget_max,
                              "q": query, "lim": limit})
            rows = cur.fetchall()
    candidates = [
        {"product_id": pid, "name": name, "brand": brand,
         "category": cat, "price": price, "summary": summary}
        for pid, name, brand, cat, price, summary in rows
    ]
    return {"candidates": candidates, "total": len(candidates)}


def search_nearest(
    database_url: str,
    query: str | None = None,
    brand: str | None = None,
    category: str | None = None,
    budget_max: float | None = None,
    limit: int = 3,
) -> dict:
    """查无结果时找「条件外最接近」的候选（防幻觉兜底，忽略预算约束）。

    排序：有预算按价格与预算差值升序；无预算按价格升序。
    """
    sql = """
        SELECT product_id, name, brand, category, price, LEFT(introduction->>0, 64)
        FROM products
        WHERE (%(cat)s::text IS NULL OR category = %(cat)s::text)
          AND (%(brand)s::text IS NULL OR brand = %(brand)s::text)
          AND (%(q)s::text IS NULL OR name ILIKE '%%' || %(q)s::text || '%%'
                                 OR brand ILIKE '%%' || %(q)s::text || '%%')
        ORDER BY CASE WHEN %(budget)s::numeric IS NULL THEN price
                      ELSE ABS(price - %(budget)s::numeric) END
        LIMIT %(lim)s
    """
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"cat": category, "brand": brand, "budget": budget_max,
                              "q": query, "lim": limit})
            rows = cur.fetchall()
    candidates = [
        {"product_id": pid, "name": name, "brand": brand,
         "category": cat, "price": price, "summary": summary}
        for pid, name, brand, cat, price, summary in rows
    ]
    return {"candidates": candidates, "total": len(candidates)}


def get_products_by_ids(database_url: str, ids: list[str]) -> dict:
    """按 ID 批量取完整商品（含 specs/introduction 全字段），缺 ID 不在结果中。"""
    if not ids:
        return {}
    sql = """
        SELECT product_id, name, brand, category, price, specs, introduction
        FROM products
        WHERE product_id = ANY(%(ids)s)
        ORDER BY product_id
    """
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"ids": ids})
            rows = cur.fetchall()
    return {
        pid: {"product_id": pid, "name": name, "brand": brand,
              "category": cat, "price": price,
              "specs": specs, "introduction": intro}
        for pid, name, brand, cat, price, specs, intro in rows
    }


def load_full_catalog(database_url: str) -> dict:
    """全量加载（mock_data 真值层快照用）：products + favorites。"""
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT product_id, name, brand, category, price, specs, introduction"
                " FROM products ORDER BY product_id"
            )
            products: dict[str, dict] = {}
            for pid, name, brand, cat, price, specs, intro in cur.fetchall():
                products[pid] = {
                    "product_id": pid, "name": name, "brand": brand,
                    "category": cat, "price": price,
                    "specs": specs, "introduction": intro,
                }
            cur.execute(
                "SELECT product_id FROM favorites ORDER BY added_at, product_id"
            )
            favorites = [row[0] for row in cur.fetchall()]
    return {"products": products, "favorites": favorites}


def init_schema_and_seed(database_url: str, catalog: dict) -> dict:
    """建表（幂等）+ 清理废弃表 + 播种（从 init_pg 调用）。

    - 废弃表 reviews / warranties 直接删除（2026-09-15 裁定不落库）
    - products 已有数据则跳过播种；favorites 为空则从本地 FAVORITES 播种
    """
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS reviews")
            cur.execute("DROP TABLE IF EXISTS warranties")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    product_id   VARCHAR PRIMARY KEY,
                    name         VARCHAR NOT NULL,
                    brand        VARCHAR NOT NULL,
                    category     VARCHAR NOT NULL,
                    price        INTEGER NOT NULL,
                    specs        JSONB NOT NULL,
                    introduction JSONB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS favorites (
                    product_id VARCHAR PRIMARY KEY REFERENCES products(product_id),
                    added_at   TIMESTAMP NOT NULL DEFAULT now()
                );
            """)
            cur.execute("SELECT count(*) FROM products")
            existing = cur.fetchone()[0]
            if not existing:
                for pid, p in catalog["products"].items():
                    cur.execute(
                        "INSERT INTO products VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (pid, p["name"], p["brand"], p["category"],
                         p["price"], json.dumps(p["specs"]),
                         json.dumps(p["introduction"])),
                    )
            cur.execute("SELECT count(*) FROM favorites")
            if cur.fetchone()[0] == 0:
                for pid in catalog["favorites"]:
                    cur.execute(
                        "INSERT INTO favorites (product_id) VALUES (%s)"
                        " ON CONFLICT (product_id) DO NOTHING",
                        (pid,),
                    )
        conn.commit()

    return {"products": len(catalog["products"]),
            "favorites": len(catalog["favorites"]),
            "existing_products": existing}
