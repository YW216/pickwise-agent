"""用户收藏：所有读写显式限定用户，迁移保留原单用户数据。"""

from app.db.connection import connect
from app.identity import validate_user_id


def init_user_schema(database_url: str, *, legacy_user_id: str = "default") -> int:
    """事务内建表并一次性导入旧收藏，返回本次导入数量。"""
    validate_user_id(legacy_user_id)
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            # 防止两个初始化进程同时建表或导入；事务结束后自动释放。
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (2026101101,))
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_favorites (
                    user_id VARCHAR(128) NOT NULL,
                    product_id VARCHAR NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
                    added_at TIMESTAMP NOT NULL DEFAULT now(),
                    PRIMARY KEY (user_id, product_id)
                );
                CREATE TABLE IF NOT EXISTS user_state_migrations (
                    name VARCHAR PRIMARY KEY,
                    applied_at TIMESTAMP NOT NULL DEFAULT now()
                );
            """)
            cur.execute(
                "INSERT INTO user_state_migrations (name) VALUES (%s)"
                " ON CONFLICT (name) DO NOTHING RETURNING name",
                ("legacy_favorites_v1",),
            )
            migrated = cur.fetchone() is not None
            imported = 0
            if migrated:
                cur.execute("SELECT to_regclass('favorites')")
                if cur.fetchone()[0] is not None:
                    cur.execute(
                        "INSERT INTO user_favorites (user_id, product_id, added_at)"
                        " SELECT %s, product_id, added_at FROM favorites"
                        " ON CONFLICT (user_id, product_id) DO NOTHING",
                        (legacy_user_id,),
                    )
                    imported = cur.rowcount
        conn.commit()
    return imported


def get_favorites(database_url: str, user_id: str, *,
                  category: str | None = None, limit: int = 20) -> list[dict]:
    """用户收藏关联商品真值；已经删除的商品不返回。"""
    validate_user_id(user_id)
    sql = """
        SELECT p.product_id, p.name, p.brand, p.category, p.price,
               LEFT(p.introduction->>0, 64)
        FROM user_favorites f JOIN products p ON p.product_id = f.product_id
        WHERE f.user_id = %(uid)s
          AND (%(cat)s::text IS NULL OR p.category = %(cat)s::text)
        ORDER BY f.added_at, f.product_id
        LIMIT %(lim)s
    """
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"uid": user_id, "cat": category, "lim": max(1, min(limit, 20))})
            rows = cur.fetchall()
    return [
        {"product_id": pid, "name": name, "brand": brand, "category": cat,
         "price": price, "summary": summary or ""}
        for pid, name, brand, cat, price, summary in rows
    ]


def add_favorite(database_url: str, user_id: str, product_id: str) -> bool:
    """幂等增加当前用户收藏；商品不存在时由外键拒绝。"""
    validate_user_id(user_id)
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_favorites (user_id, product_id) VALUES (%s, %s)"
                " ON CONFLICT (user_id, product_id) DO NOTHING",
                (user_id, product_id),
            )
            changed = cur.rowcount > 0
        conn.commit()
    return changed


def remove_favorite(database_url: str, user_id: str, product_id: str) -> bool:
    """只删除指定用户的收藏，不影响其他用户收藏同一商品。"""
    validate_user_id(user_id)
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_favorites WHERE user_id = %s AND product_id = %s",
                (user_id, product_id),
            )
            changed = cur.rowcount > 0
        conn.commit()
    return changed
