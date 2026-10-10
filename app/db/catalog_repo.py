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
        {"candidates": [brief...], "total": 返回条数, "matched": 符合条件总数}

    total 与 matched 是两个不同口径（勿混用）：
    - total   = len(candidates)，即「本次返回了几条」
    - matched = COUNT(*) OVER()，即「符合条件共几条」（不受 LIMIT 影响）
    只有 matched 才能让模型判断"结果是否被截断、要不要请用户补充条件"。
    """
    sql = """
        SELECT product_id, name, brand, category, price, LEFT(introduction->>0, 64),
               COUNT(*) OVER() AS matched
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
        for pid, name, brand, cat, price, summary, _matched in rows
    ]
    # COUNT(*) OVER() 在无匹配行时无值可取，故取不到时以返回条数兜底（此时两者必然相等：0）
    matched = rows[0][6] if rows else len(candidates)
    return {"candidates": candidates, "total": len(candidates), "matched": matched}


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


def mark_products_indexed(database_url: str, card_hashes: dict[str, str]) -> int:
    """索引成功后写回同步状态：card_hash + indexed_at（全量重建与增量同步共用）。

    - indexed_at 取数据库 now()：与 updated_at 同一 PG 时钟，脏判定
      （updated_at > indexed_at）无跨机时钟偏移问题
    - 只写这两列、不触碰业务字段 → updated_at 触发器（WHEN 业务字段）不生效，
      不会把刚同步的商品重新标脏
    - Returns: 写回的行数（应等于 len(card_hashes)）
    """
    if not card_hashes:
        return 0
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE products SET card_hash = %s, indexed_at = now()"
                " WHERE product_id = %s",
                [(h, pid) for pid, h in card_hashes.items()],
            )
        conn.commit()
    return len(card_hashes)


def init_schema_and_seed(database_url: str, catalog: dict) -> dict:
    """建表（幂等）+ 清理废弃表 + 播种（从 init_pg 调用）。

    - 废弃表 reviews / warranties 直接删除（2026-09-15 裁定不落库）
    - products 已有数据则跳过播种；favorites 为空则从本地 FAVORITES 播种
    - RAG 增量同步列（2026-10-10，develop_docs/rag增量更新.md）：
      updated_at  业务变更时间（触发器维护，仅业务字段变化时刷新）
      card_hash   检索面卡片文本的 SHA-256（判向量是否重算）
      indexed_at  上次成功写入 Milvus 的时间（NULL = 从未同步）
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
                    introduction JSONB NOT NULL,
                    updated_at   TIMESTAMP NOT NULL DEFAULT now(),
                    card_hash    VARCHAR(64),
                    indexed_at   TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS favorites (
                    product_id VARCHAR PRIMARY KEY REFERENCES products(product_id),
                    added_at   TIMESTAMP NOT NULL DEFAULT now()
                );
            """)
            # 存量库迁移（幂等）：新建库由上方 CREATE TABLE 覆盖，已有库补列
            cur.execute("""
                ALTER TABLE products
                    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT now(),
                    ADD COLUMN IF NOT EXISTS card_hash  VARCHAR(64),
                    ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMP
            """)
            # 触发器只在业务字段变化时刷新 updated_at——同步标记只写
            # card_hash / indexed_at，若被触发器连带刷 updated_at，会出现
            # updated_at > indexed_at 恒成立 → 商品永远被判脏 → 无限重算
            cur.execute("""
                CREATE OR REPLACE FUNCTION touch_products_updated_at()
                RETURNS trigger AS $fn$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $fn$ LANGUAGE plpgsql
            """)
            cur.execute("""
                DROP TRIGGER IF EXISTS products_touch_updated_at ON products
            """)
            cur.execute("""
                CREATE TRIGGER products_touch_updated_at
                BEFORE UPDATE ON products
                FOR EACH ROW
                WHEN (NEW.name          IS DISTINCT FROM OLD.name
                   OR NEW.brand         IS DISTINCT FROM OLD.brand
                   OR NEW.category      IS DISTINCT FROM OLD.category
                   OR NEW.price         IS DISTINCT FROM OLD.price
                   OR NEW.specs         IS DISTINCT FROM OLD.specs
                   OR NEW.introduction  IS DISTINCT FROM OLD.introduction)
                EXECUTE FUNCTION touch_products_updated_at()
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
