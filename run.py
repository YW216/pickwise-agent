"""服务启动入口：python run.py（代替 uvicorn app.api.app:app）。

为什么不用 uvicorn 命令行启动：事件循环策略只对"设置之后新建的循环"
生效，而 uvicorn 的顺序是先建循环、后 import 应用模块——app.py 里的
设置来不及生效，Windows 默认建出 ProactorEventLoop，psycopg 异步模式
无法工作（建连反复报错 → /api/chat 503）。

此处作为进程入口，先把策略切到 WindowsSelectorEventLoopPolicy，再交给
uvicorn 启动，保证新建的循环是 Selector 类型。host/port 与 README 一致。
"""
import asyncio
import sys

# 必须在 uvicorn 建循环之前设置：此刻进程里还没有任何事件循环
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.api.app:app", host="127.0.0.1", reload=True, port=8000)
