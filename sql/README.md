# 创建订单 · 幂等实现说明

> 本目录 (`sql/`) 存放"如果以后加下单工具"的完整示例。
> **不改动 `app/` 下任何运行时代码**——这里是设计稿，接入时按第 5 节清单改。

---

## 一、文件

| 文件 | 内容 |
|---|---|
| `01_orders_and_idempotency.sql` | 建表 DDL：`orders` + `idempotency_records` + 清理函数 |
| `test_idempotency.py` | **自包含**可跑示例：客户端 + 服务端 + 四场景验证 |

跑法：

```bash
# 建表（脚本首次运行也会自动建，这一步可选）
psql -h localhost -p 15432 -U pickwise -d pickwise -f sql/01_orders_and_idempotency.sql

# 四场景验证（含 assert，可重复跑）
/f/Anaconda/envs/searchagent/python.exe sql/test_idempotency.py
```

---

## 二、客户端 / 服务端，边界在哪

```
┌─ 客户端 ────────────────────────────────────────────┐
│ 模型侧：只输出 product_id / quantity                 │
│         ★ 看不到幂等键（所以不可能填错、也不可能每次编新的）│
│                                                      │
│ 执行层：用系统字段拼键，注入给工具                     │
│         build_idempotency_key(run_id, step, tool_call_id) │
└──────────────────────────────────────────────────────┘
                        ↓ 调用
┌─ 服务端 ────────────────────────────────────────────┐
│ 你写的 Python（test_idempotency.create_order）        │
│   ① 原子占位   ② 分支判定   ③ 首次执行 + 记账         │
│                                                      │
│ 你建的表（idempotency_records）—— 不是数据库自带       │
│                                                      │
│ PG 提供的能力（你一行都不用写）：                      │
│   唯一约束 · 行锁 · 事务 · ON CONFLICT 原子插入        │
└──────────────────────────────────────────────────────┘
```

**一句话**：幂等的"根"在数据库（唯一约束 + 原子插入），"肉"在你写的三段编排。

---

## 三、幂等键格式（定版）

```
{run_id}:{step}:{tool_call_id}
例：run-20261001-a3f2:2:call_8xQ1
```

| 段 | 作用 | 不写会怎样 |
|---|---|---|
| `run_id` | 区分"哪次用户请求" | 无法区分"用户又买一次"和"网络重试" |
| `step` | 区分"同请求的第几步" | 同一次请求里两步下单会被误判为重复 |
| `tool_call_id` | 区分"同一步的多个调用" | 一步买两件不同商品，第二个被当重复吞掉 |

**关键约束：重试必须复用同一个 `run_id`**。压缩后整体重试属于"同一次请求再跑一遍"，换个新 `run_id` 就前功尽弃。

---

## 四、核心 SQL：为什么这一行就是幂等

```sql
INSERT INTO idempotency_records (idempotency_key, tool_name, status, expires_at)
VALUES (%s, 'create_order', 'processing', now() + make_interval(hours => 24))
ON CONFLICT (idempotency_key) DO NOTHING
RETURNING idempotency_key
```

| 片段 | 作用 |
|---|---|
| `ON CONFLICT DO NOTHING` | 撞唯一约束时不报错、不做任何事 |
| `RETURNING` | 只在**真插入**时返回行 → 有值=首次，空=重复 |

**并发行为**：后到的请求在唯一索引的行锁上等待；前者提交后它发现冲突，走"已存在"分支。行锁自动完成串行化，**你不需要写任何锁代码**。

**反例（必须避免）**：
```sql
-- ❌ check-then-act：两个并发请求都查到"不存在"→ 都下单 → 幂等击穿
SELECT ... WHERE idempotency_key = %s;   -- 查到不存在
INSERT INTO idempotency_records ...;     -- 然后插入
```

---

## 五、status 三态 + 分支行为

| status | 含义 | 重复请求时的处理 |
|---|---|---|
| `processing` | 正在执行 | 报"处理中，请稍后重试"（不重跑，防并发双写） |
| `succeeded` | 已成功 | **返回缓存结果**（绝不重复下单） |
| `failed` | 已失败 | 重置为 `processing` 后重跑（允许重试） |

TTL 24 小时（`expires_at` + `cleanup_idempotency_records()`），电商下单场景足够。

---

## 六、接入 `app/` 时的改动清单

> ⚠️ 以下**尚未实施**，等你确认后再动。

### 改动 1 · `app/multi_agent/orchestrator.py` — 生成 run_id

`chat()` 里每次用户请求开始处：

```python
import uuid
run_id = uuid.uuid4().hex[:12]        # 一次用户请求 = 一个 run_id

_, new_messages = agent.handle(
    messages,
    max_steps=self.max_react_steps,
    run_id=run_id,                     # ← 新增
)
```

**重试必须复用同一个 run_id**（压缩后重试属于同一次请求）。

### 改动 2 · `app/multi_agent/agents.py` — handle 注入键

`handle()`（:74）签名加 `run_id: str = ""`；循环（:136）内：

```python
from app.agent.tools.context import (
    build_idempotency_key, set_idempotency_key, reset_idempotency_key,
)

for tc in assistant_msg.tool_calls:
    idem = build_idempotency_key(run_id, i + 1, tc.id)
    token = set_idempotency_key(idem)
    try:
        result_str = self.tool_manager.execute_call_as_message(
            tc.function.name, tc.function.arguments, seen,
        )
    finally:
        reset_idempotency_key(token)    # ← 必须 finally 复位，否则上下文污染
```

- 只读工具不读 context，注入无副作用 → 可无脑对所有调用注入
- `i + 1` 与打印的"第 i+1 步"对齐

### 改动 3 · `app/agent/tools/registry.py` — 注册工具

`_TOOL_MAP` 加 `"create_order": create_order`；`TOOL_DEFINITIONS` 加 schema：

```python
{
    "type": "function",
    "function": {
        "name": "create_order",
        "description": (
            "为用户创建订单（写操作）。仅在用户明确确认下单某商品、"
            "且已确认数量后调用。同一商品不要重复调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "商品 ID，如 LP-01", "minLength": 1},
                "quantity": {"type": "integer", "description": "购买数量，默认 1",
                             "minimum": 1, "maximum": 99},
            },
            "required": ["product_id"],
        },
    },
}
```

★ schema 里**不能出现 `idempotency_key`** —— 它由执行层注入。

### 改动 4 · `app/agent/tools/context.py` + `app/db/order_repo.py` + `app/agent/tools/order.py`

把 `test_idempotency.py` 里的三层代码拆到对应位置：
- `_idem_key` contextvars + 三个函数 → `app/agent/tools/context.py`
- `create_order` 服务端 → `app/db/order_repo.py`
- `create_order_tool` 工具层 → `app/agent/tools/order.py`（用项目统一的 `ok()/fail()` 信封）

### 改动 5 · 白名单

`agents.py` 的 `AGENT_CONFIGS`：决定哪个 Agent 有下单权限（售前给、咨询一般不给）。
