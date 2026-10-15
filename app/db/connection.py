"""PG 连接层：同步 connect（脚本/启动路径）+ 异步连接池（chat 链路查询）。

异步池按 ``database_url`` 多实例缓存（bind_tool_handlers 支持同进程多个
ToolManager 指向不同库），惰性创建：首个查询触发建池，服务退出统一关闭。
"""
import asyncio
import sys

import psycopg
from psycopg_pool import AsyncConnectionPool

# Windows 上 psycopg 异步模式要求 Selector 事件循环（Proactor 不支持
# add_reader）。所有使用异步池的入口（工具测试直调、脚本）import 本模块
# 即生效，与 app.py 的同款设置互为双保险。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# url → 池。检查-创建之间存在 await（pool.open），用锁保证单例。
_pools: dict[str, AsyncConnectionPool] = {}
_pool_lock = asyncio.Lock()


def connect(database_url: str):
    """打开同步连接（脚本/启动/维护路径，调用方负责 with/close）。"""
    return psycopg.connect(database_url)


def ping(database_url: str) -> tuple[bool, str]:
    """同步连通性探测（快败，3 秒超时；DependencyProbe 走线程池调用）。"""
    try:
        with psycopg.connect(database_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


async def _pool_for(database_url: str) -> AsyncConnectionPool:
    """按 URL 取异步池，不存在则创建并后台填充连接（double-check 单例）。"""
    pool = _pools.get(database_url)
    if pool is not None:
        return pool
    #防止并发重复建锁
    async with _pool_lock:
        pool = _pools.get(database_url)
        if pool is not None:
            return pool
        from app.config.settings import settings

        pool = AsyncConnectionPool(
            conninfo=database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            open=False,
        )
        await pool.open()  # 后台初始化：PG 不可达不阻塞（§4.6），状态由 /api/ready 暴露
        _pools[database_url] = pool
        return pool


async def close_all_pools() -> None:
    """服务退出时统一关闭全部异步池（lifespan 调用）。"""
    for pool in list(_pools.values()):
        await pool.close()
    _pools.clear()
