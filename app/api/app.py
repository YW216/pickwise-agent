"""FastAPI 应用工厂（多租户隔离和fastapi接入.md §4.1/§4.6）。

生产入口 `app.api.app:app`。lifespan 在启动时创建 SessionManager 与
有界执行器，退出时按依赖逆序关闭；同一个工厂创建的每个 app 实例拥有
独立的运行对象（测试隔离），绝不共享全局单例。
"""

from __future__ import annotations

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Windows 上 psycopg 异步模式要求 Selector 事件循环（Proactor 不支持
# add_reader）；import 本模块即生效，任何启动方式都覆盖。uvicorn 在
# Windows + asyncio 下也会做同样设置，此处是双保险。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.api import routes
from app.api.auth import AnonymousAuth
from app.api.envelope import (
    ProductBusy,
    MAX_BODY_BYTES,
    error,
    new_request_id,
)
from app.config.settings import settings
from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.services.async_runtime import AsyncRuntime
from app.services.session_manager import SessionManager


class ProductRunner:
    """商品/收藏接口的有界执行器：名额满立即拒绝，不排队（§4.1）。"""

    def __init__(self, max_workers: int):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="product-worker",
        )
        self._slots = asyncio.Semaphore(max_workers)

    async def run(self, fn, *args):
        if self._slots.locked():  # 四个名额都在使用：直接 503，不增加等待队列
            raise ProductBusy()
        async with self._slots:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, fn, *args)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class DependencyProbe:
    """依赖探测：短超时 + 结果缓存（健康请求不触发付费调用，§4.6）。"""

    def __init__(self, cache_seconds: float = 30.0):
        self._cache_seconds = cache_seconds
        self._cached: dict | None = None
        self._cached_at = 0.0
        self._lock = asyncio.Lock()

    async def snapshot(self) -> dict:
        async with self._lock:
            now = time.monotonic()
            if self._cached is not None and now - self._cached_at < self._cache_seconds:
                return self._cached
            self._cached = await self._probe()
            self._cached_at = now
            return self._cached

    async def _probe(self) -> dict:
        loop = asyncio.get_running_loop()
        pg_ok = await self._try(loop, self._probe_pg, timeout=3.0)
        milvus_ok = await self._try(loop, self._probe_milvus, timeout=3.0)
        return {
            "ready": bool(pg_ok),  # 核心依赖：结构化商品与收藏（PG）
            "degraded": bool(pg_ok and not milvus_ok),
            "dependencies": {
                "postgres": "ok" if pg_ok else "unavailable",
                # Milvus 的 TCP 可达 ≠ 索引可查询；此处只标连通级别，
                # 索引健康由真实检索调用的结果反映（不在此做付费验证）
                "milvus": "ok" if milvus_ok else "degraded",
            },
            "verified": ["local-probe"],  # 标注：远程模型 API 可用性未在探测中验证
        }

    async def _try(self, loop, fn, *, timeout: float) -> bool:
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, fn), timeout)
        except Exception:  # noqa: BLE001 —— 探测失败即不可用，不向上抛
            return False

    @staticmethod
    def _probe_pg() -> bool:
        if not settings.database_url:
            return False
        from app.db.connection import ping
        ok, _ = ping(settings.database_url)
        return ok

    @staticmethod
    def _probe_milvus() -> bool:
        if not settings.milvus_uri:
            return False
        from app.agent.rag.milvus_utils import ensure_reachable
        try:
            ensure_reachable(settings.milvus_uri)
            return True
        except Exception:  # noqa: BLE001
            return False


class _PayloadTooLarge(Exception):
    """实际到达字节超过上限（接收中途判定）。"""


class BodySizeLimitMiddleware:
    """请求体大小上限：Content-Length 预检 + 实际接收字节校验（§4.4）。"""

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "DELETE"}:
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length and int(content_length) > self.max_bytes:
            # 声明就超限：不读 body，直接 413
            await error(413, "REQUEST_TOO_LARGE",
                        f"请求体超过 {self.max_bytes} 字节上限",
                        new_request_id())(scope, receive, send)
            return

        received = 0

        async def receive_wrapper():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _PayloadTooLarge()  # 中断读取（无 Content-Length 的兜底）
            return message

        try:
            await self.app(scope, receive_wrapper, send)
        except _PayloadTooLarge:
            await error(413, "REQUEST_TOO_LARGE",
                        f"请求体超过 {self.max_bytes} 字节上限",
                        new_request_id())(scope, receive, send)


def create_app(*, auth_secret_key: str | None = None,
               session_dir: str | Path | None = None,
               factory=None) -> FastAPI:
    """创建 FastAPI 应用；三个注入参数仅供测试，生产读配置。

    factory：identity → 运行对象。默认构造真实 MultiAgentOrchestrator；
    测试必须注入假 factory，绝不让单元测试打到真实模型。
    """
    injected = auth_secret_key is not None
    secret = auth_secret_key if injected else settings.auth_secret_key
    if not injected:
        settings.assert_auth_configured()  # 配置错误阻止启动（§4.6）
    if len(secret.encode()) < 32:
        raise RuntimeError("AUTH_SECRET_KEY 过弱：至少 32 字节高强度随机材料")
    orchestrator_factory = factory or (
        # 所有窗口的 orchestrator 复用共享 AsyncOpenAI（连接池共享）；
        # 模型并发限流已移除（阶段八裁定，自用场景无对手方）
        lambda identity: MultiAgentOrchestrator(
            identity=identity,
            client=app.state.async_runtime.model_client))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动顺序：配置校验（上方）→ 异步基础层 → SessionManager/执行器 → 清理任务（§4.6）
        app.state.auth = AnonymousAuth(secret, settings.auth_cookie_max_age_days * 86400)
        app.state.async_runtime = AsyncRuntime()
        await app.state.async_runtime.open()  # 异步模型客户端 + PG 连接池
        app.state.session_manager = SessionManager(
            session_dir=Path(session_dir or settings.web_session_dir),
            factory=orchestrator_factory,
        )
        app.state.product_runner = ProductRunner(settings.product_executor_workers)
        app.state.dependency_probe = DependencyProbe()
        await app.state.session_manager.start_cleanup_loop()
        try:
            yield
        finally:
            # 退出逆序：SessionManager（排空并逐窗口保存关闭）→ 异步基础层 → 执行器 → 公共客户端
            await app.state.session_manager.close()
            await app.state.async_runtime.aclose()
            from app.db.connection import close_all_pools
            await close_all_pools()
            app.state.product_runner.shutdown()
            from app.agent.tools.search_products import close_public_clients
            close_public_clients()

    app = FastAPI(title="PickWise", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,  # 显式配置；credentials 下不允许通配符
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(routes.auth_router)
    app.include_router(routes.sessions_router)
    app.include_router(routes.chat_router)
    app.include_router(routes.products_router)
    app.include_router(routes.favorites_router)
    app.include_router(routes.health_router)
    return app


def __getattr__(name: str):
    """生产入口惰性创建（uvicorn app.api.app:app）。

    与 settings.py 同一设计哲学：import 不触发密钥校验——无 .env 的
    离线单测照常收集，真正启动服务时才校验并给出可操作的报错。
    """
    if name == "app":
        module_globals = globals()
        if module_globals.get("app") is None:
            module_globals["app"] = create_app()
        return module_globals["app"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
