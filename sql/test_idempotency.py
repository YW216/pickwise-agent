"""幂等到手示例：客户端 + 服务端完整可跑（2026-10-01，自包含）。

跑法（项目根目录）：
    /f/Anaconda/envs/searchagent/python.exe sql/test_idempotency.py

先建表（本脚本首次运行时会自动建，无需手动）：
    sql/01_orders_and_idempotency.sql

============================================================================
这个文件把"客户端"和"服务端"都写全了，对照着看：
============================================================================

【客户端】= 模型侧 + 执行层
  - 模型侧：只给 product_id / quantity，**看不到幂等键**（见 create_order_tool 的注释）
  - 执行层：用系统字段拼幂等键并"注入"给工具（见 IdempotencyContext）

【服务端】= 你写的 Python（本文件的 create_order 函数）+ 你建的表 + PG 的能力
  - create_order 负责编排三段：① 原子占位 ② 分支判定 ③ 首次执行+记账
  - idempotency_records 表由你建（不是数据库自带）
  - PG 提供：唯一约束、行锁、事务、ON CONFLICT 原子插入（这些你一行都不用写）
"""

import json
import sys
import threading
from contextvars import ContextVar
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

from app.config.settings import settings


# ===========================================================================
# 第 0 层：幂等键的注入通道（contextvars）
#
# 为什么需要它：幂等键必须跨重试保持一致，只能由执行层用系统字段拼；
# 而工具函数签名里又不能出现这个参数（出现了模型就会尝试填它）。
# 两头一夹，就需要"执行层写、工具读、模型不可见"的通道；
# 用 contextvars 而非全局变量，是为了并发执行时自动隔离、不串号。
# ===========================================================================
_idem_key: ContextVar[str | None] = ContextVar("idempotency_key", default=None)


def build_idempotency_key(run_id: str, step: int, tool_call_id: str) -> str:
    """按约定格式拼幂等键：{run_id}:{step}:{tool_call_id}。

    三段缺一不可：
      run_id       区分"哪次用户请求"（重试时不变 → 才能命中同一条记录）
      step         区分"同一请求的第几步"
      tool_call_id 区分"同一步里的多个调用"（模型可能一步发多个下单）
    """
    return f"{run_id}:{step}:{tool_call_id}"


# ===========================================================================
# 第 1 层：工具函数（客户端看到的那个"工具"）
#
# 注意签名 —— 只有 product_id 和 quantity，**没有 idempotency_key**。
# 这是刻意的设计：键必须由系统生成，不能交给模型（模型每次可能编不同的值）。
# ===========================================================================
def create_order_tool(product_id: str, quantity: int = 1) -> dict:
    """创建订单工具。参数与 OpenAI function schema 一致，键从上下文读。"""
    key = _idem_key.get()
    if not key:
        return {"success": False, "error": "下单缺少幂等上下文，请通过对话重新发起"}

    try:
        payload = create_order(
            database_url=settings.database_url,
            idempotency_key=key,
            user_id=settings.memory_user_id,
            product_id=product_id,
            quantity=quantity,
        )
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    msg = (
        f"该订单此前已创建成功，未重复下单。订单号 {payload['order_id']}，"
        f"共 {payload['total_price']} 元。"
        if payload["replayed"]
        else f"下单成功。订单号 {payload['order_id']}，"
             f"{payload['quantity']} 件，共 {payload['total_price']} 元。"
    )
    return {"success": True, "error": None, "data": {**payload, "message": msg}}


# ===========================================================================
# 第 2 层：服务端 —— 这三段就是幂等的全部实现
# ===========================================================================
TTL_HOURS = 24


def create_order(
    database_url: str,
    idempotency_key: str,
    user_id: str,
    product_id: str,
    quantity: int = 1,
) -> dict:
    """下单（幂等）。服务端核心。

    幂等语义（四种情况）：
      A. 首次调用           → 占位成功 → 执行下单 → 记账 succeeded → 返回新订单
      B. 重复（已成功）     → 占位冲突 → 读到 succeeded → 返回缓存结果（不重下）
      C. 重复（执行中）     → 占位冲突 → 读到 processing → 报"处理中请稍后"
      D. 重复（曾失败）     → 占位冲突 → 读到 failed → 重置为 processing 重跑

    Raises:
        ValueError：商品不存在 / 数量非法 / 处理中（业务失败，工具层转成 fail）
    """
    if quantity < 1:
        raise ValueError("下单数量必须大于 0")

    # with conn: 就是事务边界 —— 正常退出 commit，抛异常 rollback。
    # 下单（写 orders）与记账（写 idempotency_records）必须同生共死：
    #   订单成功但记账失败 → key 永远卡 processing，后续重试全被拒
    #   记账成功但订单没写 → 重复请求会拿到"成功"的假结果
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            # ---- ① 原子占位：一行 SQL 完成"判断 + 写入" --------------------
            # ON CONFLICT DO NOTHING：撞到 idempotency_key 唯一约束时不做任何事
            # RETURNING：只在**真插入新行**时返回行；冲突时返回空
            #   → 返回了行 = 首次；返回空 = 已存在（重复请求）
            # 两个并发请求同时到达时，后到的那个在行锁上等待，等前一个落定后
            # 发现冲突，走"已存在"分支 —— 天然串行化，不需要你手动加锁。
            # 绝不能写成"先 SELECT 再 INSERT"：那是 check-then-act，
            # 并发下两个请求都会查到"不存在"然后都下单，幂等直接击穿。
            cur.execute(
                """
                INSERT INTO idempotency_records
                    (idempotency_key, tool_name, status, expires_at)
                VALUES (%s, 'create_order', 'processing',
                        now() + make_interval(hours => %s))
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING idempotency_key
                """,
                (idempotency_key, TTL_HOURS),
            )
            first_time = cur.fetchone() is not None

            # ---- ② 重复请求：读旧状态，决定"重放"还是"重试" ----------------
            if not first_time:
                cur.execute(
                    "SELECT status, result FROM idempotency_records"
                    " WHERE idempotency_key = %s",
                    (idempotency_key,),
                )
                status, result = cur.fetchone()

                if status == "succeeded":
                    # 情况 B：已经成功过 → 原样返回，绝不重复下单
                    return {**result, "replayed": True}
                if status == "processing":
                    # 情况 C：另一个请求正在处理 → 让调用方稍后重试
                    raise ValueError("该下单请求正在处理中，请稍后重试")
                # 情况 D：上次失败 → 重置为 processing，落到下面重跑
                cur.execute(
                    "UPDATE idempotency_records"
                    " SET status = 'processing', updated_at = now()"
                    " WHERE idempotency_key = %s",
                    (idempotency_key,),
                )

            # ---- ③ 首次执行：查商品 → 写订单 → 记账 ------------------------
            cur.execute(
                "SELECT price FROM products WHERE product_id = %s", (product_id,),
            )
            row = cur.fetchone()
            if row is None:
                # 业务失败也要记账，否则这条 key 永远卡在 processing
                cur.execute(
                    "UPDATE idempotency_records SET status = 'failed',"
                    " updated_at = now() WHERE idempotency_key = %s",
                    (idempotency_key,),
                )
                raise ValueError(f"商品不存在: {product_id}")

            unit_price = row[0]
            total_price = unit_price * quantity
            cur.execute("SELECT to_char(now(), 'YYYYMMDD')")
            day = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM orders WHERE order_id LIKE %s",
                (f"ORD-{day}-%",),
            )
            order_id = f"ORD-{day}-{cur.fetchone()[0] + 1:04d}"

            cur.execute(
                "INSERT INTO orders (order_id, user_id, product_id, quantity,"
                " unit_price, total_price, status)"
                " VALUES (%s, %s, %s, %s, %s, %s, 'created')",
                (order_id, user_id, product_id, quantity, unit_price, total_price),
            )

            payload = {
                "order_id": order_id, "user_id": user_id, "product_id": product_id,
                "quantity": quantity, "unit_price": unit_price,
                "total_price": total_price, "status": "created",
            }
            cur.execute(
                "UPDATE idempotency_records SET status = 'succeeded',"
                " result = %s, updated_at = now() WHERE idempotency_key = %s",
                (json.dumps(payload), idempotency_key),
            )

        # with conn 退出 = commit：orders 与 idempotency_records 一起落盘
        return {**payload, "replayed": False}


# ===========================================================================
# 第 3 层：建表（本项目先例：app/db/catalog_repo.init_schema_and_seed）
# ===========================================================================
def ensure_tables(database_url: str) -> None:
    """建表（幂等）+ 保证有可下单的商品。"""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id    VARCHAR PRIMARY KEY,
                    user_id     VARCHAR NOT NULL,
                    product_id  VARCHAR NOT NULL,
                    quantity    INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
                    unit_price  INTEGER NOT NULL CHECK (unit_price >= 0),
                    total_price INTEGER NOT NULL CHECK (total_price >= 0),
                    status      VARCHAR NOT NULL DEFAULT 'created',
                    created_at  TIMESTAMP NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_orders_user
                    ON orders (user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS idempotency_records (
                    idempotency_key VARCHAR(128) PRIMARY KEY,
                    tool_name       VARCHAR(64)  NOT NULL,
                    status          VARCHAR(16)  NOT NULL,
                    result          JSONB,
                    error           TEXT,
                    created_at      TIMESTAMP NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMP NOT NULL DEFAULT now(),
                    expires_at      TIMESTAMP NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_idem_expires
                    ON idempotency_records (expires_at);
            """)
            # 演示用兜底商品：真项目里 products 由 init_pg 播种
            cur.execute(
                "INSERT INTO products (product_id, name, brand, category, price,"
                " specs, introduction)"
                " VALUES ('LP-01', '凌霄 Pro 14', '星海', '笔记本', 6999, '{}', '[]')"
                " ON CONFLICT (product_id) DO NOTHING"
            )
        conn.commit()


def _count_orders(url: str, product_id: str) -> int:
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM orders WHERE product_id = %s", (product_id,))
            return cur.fetchone()[0]


# ===========================================================================
# 演示：四种情况
# ===========================================================================
def main():
    url = settings.database_url
    if not url:
        print("未配置 database_url"); return

    ensure_tables(url)
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM idempotency_records WHERE idempotency_key LIKE 'demo:%'")
            cur.execute("DELETE FROM orders")
        conn.commit()

    pid = "LP-01"
    print("=" * 72)

    # ---- A. 首次 ----
    print("\n【A】首次下单   key = demo:A")
    tok = _idem_key.set("demo:A")
    r1 = create_order_tool(pid, 2)
    _idem_key.reset(tok)
    p1 = r1["data"]
    print(f"    order_id={p1['order_id']}  总额={p1['total_price']}  replayed={p1['replayed']}")
    print(f"    orders 表行数 = {_count_orders(url, pid)}")

    # ---- B. 同键重放 ----
    print("\n【B】同键再来一次   key = demo:A（模拟网络超时后客户端重试）")
    tok = _idem_key.set("demo:A")
    r2 = create_order_tool(pid, 2)
    _idem_key.reset(tok)
    p2 = r2["data"]
    print(f"    order_id={p2['order_id']}  replayed={p2['replayed']}")
    print(f"    orders 表行数 = {_count_orders(url, pid)}   ← 没有变多！")
    assert p1["order_id"] == p2["order_id"], "同键必须返回同一订单"
    assert p2["replayed"] is True

    # ---- C. 换键 = 新下单 ----
    print("\n【C】换个键   key = demo:C（模拟用户真的又买了一次）")
    tok = _idem_key.set("demo:C")
    r3 = create_order_tool(pid, 1)
    _idem_key.reset(tok)
    print(f"    order_id={r3['data']['order_id']}  replayed={r3['data']['replayed']}")
    print(f"    orders 表行数 = {_count_orders(url, pid)}   ← 这次真的多了一单")
    assert r3["data"]["replayed"] is False

    # ---- D. 并发同键 ----
    print("\n【D】并发 5 个请求，全部用同一个 key = demo:D")
    results, errors = [], []
    def worker():
        tok = _idem_key.set("demo:D")
        try:
            results.append(create_order_tool(pid, 1))
        except Exception as exc:
            errors.append(str(exc))
        finally:
            _idem_key.reset(tok)
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

    ok_list = [r for r in results if r["success"]]
    fresh = [r for r in ok_list if not r["data"]["replayed"]]
    replay = [r for r in ok_list if r["data"]["replayed"]]
    print(f"    真下单={len(fresh)}  重放={len(replay)}  报错={len(errors)}")
    if errors:
        print(f"    重试提示：{errors}")
    print(f"    orders 表行数 = {_count_orders(url, pid)}（含前面 A/C 两单）")
    # 核心断言：无论并发多凶，真下单只能有一次
    assert len(fresh) <= 1, f"幂等被击穿！真下单 {len(fresh)} 次"

    print("\n" + "=" * 72)
    print("结论：同一个幂等键，无论重复几次、并发多高，orders 表只会多一行。")
    print("=" * 72)


if __name__ == "__main__":
    main()
