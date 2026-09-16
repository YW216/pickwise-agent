"""PG 连接与连通性探测（同步 psycopg3）。"""
import psycopg


def connect(database_url: str):
    """打开连接（调用方负责 with/close）。"""
    return psycopg.connect(database_url)


def ping(database_url: str) -> tuple[bool, str]:
    """连通性探测（快败，3 秒超时）。"""
    try:
        with psycopg.connect(database_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
