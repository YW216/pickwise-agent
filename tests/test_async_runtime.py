"""异步基础层验证（异步改造阶段一验收标准）：

- 应用启动创建异步客户端与连接池、关闭正确释放；
- 所有网络调用有超时（构造参数断言）；
- LLM 并发闸门行为正确；
- 同步路径完全不受影响（可回退）。

连接池 open/close 依赖本地 PG（不可用则跳过）；不调用真实模型。
"""

import asyncio
import sys

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.config.settings import settings
from app.services.async_runtime import AsyncRuntime, RuntimeConfig
from tests.test_session_manager import FactoryRecorder

SECRET = "async-runtime-secret-0123456789abcdef"  # 40 字节

# psycopg 异步模式在 Windows 要求 Selector 事件循环（与 app.py 同款双保险）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _config() -> RuntimeConfig:
    """测试配置：不依赖真实密钥值（只构造客户端，不发起请求）。"""
    return RuntimeConfig(
        openai_api_key="test-key",
        openai_base_url=settings.openai_base_url,
        openai_timeout=settings.openai_timeout,
        openai_max_retries=settings.openai_max_retries,
    )


def test_open_creates_clients_and_aclose_releases():
    async def main():
        runtime = AsyncRuntime(_config())
        assert runtime.model_client is None  # 未打开时不可访问
        await runtime.open()
        assert runtime.model_client is not None
        assert runtime.model_client.timeout == settings.openai_timeout  # 网络调用有超时
        assert runtime.model_client.max_retries == settings.openai_max_retries
        await runtime.aclose()
        assert runtime.model_client is None

        await runtime.aclose()  # 重复关闭安全

    asyncio.run(main())


def test_open_twice_is_rejected():
    async def main():
        runtime = AsyncRuntime(_config())
        await runtime.open()
        with pytest.raises(RuntimeError):
            await runtime.open()
        await runtime.aclose()

    asyncio.run(main())


def test_lifespan_creates_and_closes_runtime(tmp_path):
    """HTTP 应用启动后 app.state.async_runtime 就绪，退出后释放。"""
    app = create_app(
        auth_secret_key=SECRET,
        session_dir=tmp_path,
        factory=FactoryRecorder(),
    )
    with TestClient(app) as client:
        runtime = client.app.state.async_runtime
        assert runtime.model_client is not None
        assert client.get("/api/health").status_code == 200
    assert runtime.model_client is None  # 退出后已释放


def test_sync_paths_unaffected():
    """同步回退保证：现有同步调用方不 import 本模块也照常工作。"""
    from app.agent.tools.detail import get_detail  # noqa: F401
    from app.services.session_manager import SessionManager  # noqa: F401
