"""Milvus 连接辅助：TCP 层快败探活。

背景：gRPC 惰性连接 + pymilvus 内部重试策略，会把"服务没启动"拖成
数分钟假死；且 pymilvus 的 timeout 只截断单次 RPC，超时异常会被它自己的
重试层反复重连——所以探活必须在 TCP 层做，不经过 pymilvus。
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse


def ensure_reachable(uri: str, timeout: float = 2.0) -> None:
    """对 standalone 的 http uri 做 TCP 探活，不可达时立刻抛人话 ConnectionError。"""
    probe = urlparse(uri)
    host, port = probe.hostname, probe.port or 19530
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as e:
        raise ConnectionError(
            f"Milvus 服务不可达（{host}:{port}）。"
            "standalone 请先启动：deploy/milvus 下 docker compose up -d，"
            "并等 docker compose ps 里 standalone 变为 healthy。"
        ) from e
