"""商品索引增量同步 worker：独立进程，定期把 PG 真值同步进 Milvus。

设计（多租户总方案 5.6，承接 rag增量更新.md 第 5 步）：
- 与 backend 同代码不同进程（``python -m app.worker``），compose 部署时
  双 service 并存；本地开发直接跑本模块即可
- "跑完再计时"防重入：delay 从本轮完成时刻起算，慢轮次不会堆积
- 单轮失败不退出：退避重试（PG/Milvus 短暂不可达时自动恢复）
- SIGTERM（Linux）优雅退出，当前轮跑完为止；Windows 用 Ctrl+C 退出

service 层零改动：sync() 的任何失败抛异常，由本外壳决定如何呈现。
"""

from __future__ import annotations

import signal
import sys
import time

from app.config.settings import settings
from app.services.product_sync import sync

_stop = False


def _on_sigterm(signum, frame) -> None:  # pragma: no cover - 信号回调
    """SIGTERM → 当前轮跑完后退出（优雅关闭）。"""
    global _stop
    _stop = True


def run_worker_loop(stop_event=None) -> None:
    """同步循环本体（两种伴生方式共用）：

    - 独立进程：main() 直接调它，SIGTERM 置 _stop
    - FastAPI 伴生：daemon 线程调它，stop_event 为 threading.Event
    """
    interval = settings.worker_interval_seconds
    retry = settings.worker_retry_seconds
    print(f"[worker] 增量同步启动：每 {interval}s 一轮，失败退避 {retry}s", flush=True)

    while not _stop and not (stop_event is not None and stop_event.is_set()):
        started = time.monotonic()
        try:
            stats = sync()
            cost = time.monotonic() - started
            print(f"[worker] 同步完成：{stats}（耗时 {cost:.1f}s）", flush=True)
            delay = interval
        except KeyboardInterrupt:
            break
        except Exception as exc:  # noqa: BLE001 —— worker 不因单轮失败退出
            print(f"[worker] 同步失败，{retry}s 后重试：{type(exc).__name__}: {exc}", flush=True)
            delay = retry
        # sleep 分段短睡，让退出信号能及时打断长退避
        waited = 0.0
        should_stop = lambda: _stop or (stop_event is not None and stop_event.is_set())
        while waited < delay and not should_stop():
            step = min(1.0, delay - waited)
            time.sleep(step)
            waited += step

    print("[worker] 已退出", flush=True)


def main() -> None:
    """独立进程入口（python -m app.worker）：SIGTERM/Ctrl+C 退出。"""
    if hasattr(signal, "SIGTERM") and sys.platform != "win32":
        signal.signal(signal.SIGTERM, _on_sigterm)
    run_worker_loop()


if __name__ == "__main__":
    main()
