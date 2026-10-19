"""服务启动入口：python run.py（代替 uvicorn app.api.app:app）。

为什么不用 uvicorn 命令行启动：事件循环策略只对"设置之后新建的循环"
生效，而 uvicorn 的顺序是先建循环、后 import 应用模块——app.py 里的
设置来不及生效，Windows 默认建出 ProactorEventLoop，psycopg 异步模式
无法工作（建连反复报错 → /api/chat 503）。

此处作为进程入口，先把策略切到 WindowsSelectorEventLoopPolicy，再交给
uvicorn 启动，保证新建的循环是 Selector 类型。host/port 与 README 一致。

启动前先检查端口：Windows 允许两个进程绑同一端口，热重载残留的旧进程会
把请求接走，表现为「接口全超时」或「client has been closed」，排查成本很高，
所以这里直接拒绝启动并给出清理命令。
"""
import asyncio
import socket
import sys

# 必须在 uvicorn 建循环之前设置：此刻进程里还没有任何事件循环
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

HOST = "127.0.0.1"
PORT = 8000

CLEANUP_HINT = (
    "  清理旧进程（PowerShell）：\n"
    "    Get-NetTCPConnection -LocalPort {port} -State Listen |"
    " ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force }}"
).format(port=PORT)


def port_in_use(host: str, port: int) -> bool:
    """能连上就说明已有监听者；超时/拒绝都视为空闲。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


if __name__ == "__main__":
    if port_in_use(HOST, PORT):
        print(f"端口 {PORT} 已被占用，本次启动取消。\n{CLEANUP_HINT}")
        sys.exit(1)
    # reload_delay 给旧 worker 留出退出时间，减少 Windows 上双进程抢端口。
    uvicorn.run(
        "app.api.app:app",
        host=HOST,
        port=PORT,
        reload=True,
        reload_delay=1.0,
    )
