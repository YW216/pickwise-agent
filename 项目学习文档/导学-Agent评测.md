# 导学 · Agent 评测（PickWise / ecom-service-agent-main）

> **本文解决什么问题**：从头到尾讲清"agent 评测是怎么设计的、每部分为什么这样写、代码在哪一行"，重点补齐 **Langfuse 的概念、接入方式与"我们到底加了什么"**。
>
> **三份文档的分工**（不要混读）：
> - `项目学习文档/导学-PickWise.md` —— 全局学习路线（本文是它的**第 3 站深化**）。
> - `develop_docs/模块设计/11-评估.md` —— **模块设计文档**（为什么这么设计，决策与取舍）。
> - `develop_docs/学习笔记/agent评测过程.md` —— **问题演进记录**（今天踩了哪些坑、怎么修的、前后对照）。
> - **本文** —— **学习理解文档**：以"读代码"为纲，把设计、关联、Langfuse 概念一次性讲透。
>
> 全部行号基于 2026-09-16 源码实测，可直接在仓库定位。

## 0. 先建立全局地图

在深入任何细节前，先记住这张图——后面每一节都是在展开它其中的一个方块。

```
  ① 用例集                        ② 采集（沙箱）                  ③ 判分                    ④ 呈现
┌────────────────┐          ┌──────────────────────┐      ┌──────────────┐      ┌─────────────────┐
│ cases.json     │          │ Sandbox              │      │ metrics.py   │      │ 终端报告         │
│ (EvalCase ×20) │─────────>│  · 隔离环境           │─────>│ 9 条规则断言  │─────>│ runs/*.json     │
│ router_cases   │          │  · 插桩（monkeypatch）│      │ judges.py    │      │ report.json     │
│ (RouterCase×55)│  多轮输入 │  · 产出 RunTrace      │      │ LLM judge    │      │ Langfuse 看板    │
└────────────────┘          └──────────────────────┘      └──────────────┘      └─────────────────┘
                                       │                        │                     ▲
                                       └──── RunTrace（唯一数据载体）┘              │
                                                          reporter.py ───────────────┘
```

关键设计意图一句话：**采集与评分彻底解耦**——沙箱只负责"把 Agent 跑一遍并记录事实"，判分器只读 `RunTrace`，两者互不知道对方存在。这样加断言不用改沙箱，换采集方式不用改判分。

### 0.1 代码目录速查

| 路径                                 | 职责                                     | 本文对应章节      |
| ---------------------------------- | -------------------------------------- | ----------- |
| `app/evaluation/dataset.py`        | 端到端用例数据结构 + 加载                         | §3.1        |
| `app/evaluation/cases.json`        | 20 条端到端用例                              | §3.1        |
| `app/evaluation/sandbox.py`        | 隔离 + 插桩 + 执行                           | §3.2 / §3.3 |
| `app/evaluation/trace.py`          | `RunTrace` 数据载体                        | §3.4        |
| `app/evaluation/metrics.py`        | 9 条确定性规则断言                             | §3.5        |
| `app/evaluation/judges.py`         | LLM judge（advisory）                    | §3.6        |
| `app/evaluation/evaluator.py`      | 编排：跑用例 → 判分 → 聚合                       | §3.7        |
| `app/evaluation/reporter.py`       | **Langfuse 上报（全部接入点）**                 | §5          |
| `app/evaluation/router_dataset.py` | Router 用例结构 + 加载                       | §4          |
| `app/evaluation/router_cases.json` | 55 条 Router 用例                         | §4          |
| `app/scripts/run_eval.py`          | 端到端入口（L3）                              | §3.7        |
| `app/scripts/run_router_eval.py`   | Router 入口（L1）                          | §4          |
| `app/scripts/run_rag_eval.py`      | RAG 入口（L2）                             | §2          |
| `app/evaluation/runs/`             | 落盘产物（`<case_id>.json` + `report.json`） | §7          |
| `tests/test_evaluation.py`         | 评测模块自身的单测（38 个）                        | §3.5 / §9   |

### 0.2 一句话理解三层

| 层             | 被测对象                               | 用例数 | 判分方式                             | 入口                   | 回答什么问题         |
| ------------- | ---------------------------------- | --- | -------------------------------- | -------------------- | -------------- |
| **L1 Router** | `Router.route()` 单个方法              | 55  | 输出与期望场景列表**完全一致**                | `run_router_eval.py` | 意图分类准不准        |
| **L2 RAG**    | 检索器（知识库 + 商品库）                     | 152 | Hit@K / MRR / Recall / Precision | `run_rag_eval.py`    | 检索召得全不全、排得准不准  |
| **L3 端到端**    | 完整 `MultiAgentOrchestrator.chat()` | 20  | 9 条规则断言 + LLM judge              | `run_eval.py`        | 整个商品推荐助手到底好不好用 |

> **为什么必须分层**：成本与归因粒度成反比。L1 不改产品代码、单条几秒，能精确到"哪条规则挂了"；L3 真跑 Agent、单条要十几次 LLM 调用，但能发现"路由对了、检索也对了，可最终回复还是编了个价格"这种**只有端到端才暴露**的问题。层与层不是替代关系，是**漏斗**关系。

## 1. 为什么需要评测：先定义问题

这一节的目的是让你理解——**后面每一个设计决策，都是被某个具体问题逼出来的**。

**问题**：改了一句 `PRESALE_PROMPT`、换了模型版本、调了 `max_tokens`，系统到底是变好了还是变差了？靠人肉聊几句？聊 3 个 case 就说"好像还行"？——这就是"发展快、质量不可控"的根源。

从这个问题推导出三条**硬约束**，三条约束又各自逼出一个设计：

| 约束 | 为什么 | 逼出的设计 | 代码落点 |
| --- | --- | --- | --- |
| **确定性** | 同输入同输出，才有"回归"可言；LLM 打分本身会抖 | 通过门槛只用**规则断言**（零 LLM）；judge 只做 advisory | `metrics.py` 全部函数 |
| **隔离** | 跑评测不能读真实用户记忆、不能改真实 session、不能依赖外网 MCP | 每条用例独立临时 session + 独立 memory_dir + 关 MCP | `sandbox.py:48-62` |
| **可归因** | 失败了要知道"是哪一层、哪个工具的锅" | 插桩采集全量轨迹（token / 工具入参出参 / 每轮路由 / finish_reason） | `sandbox.py:98-225` |

还有一个隐含约束：**不能为了评测改产品代码**。产品代码里塞 `if is_eval: log(...)` 是最脏的做法——会污染主链路、增加分支、还得随重构同步维护。所以采集必须走**插桩**（§3.3）。这条约束直接决定了 `sandbox.py` 的整个形态。

## 2. 三层评测体系与职责边界

### 2.1 分层不是"三个脚本"，是三种粒度的证据

- **L1 Router 评测**（`run_router_eval.py`）——**最便宜的定点验证**。它甚至不建 Agent，直接 `Router(client, model)` 然后逐条 `route()`（`run_router_eval.py:38-49`）。改 `ROUTER_PROMPT` 后必须跑。
- **L2 RAG 评测**（`run_rag_eval.py`）——**检索质量专项**。用 Hit@K / MRR 这类检索学标准指标，分 narrow / wide / dense 三种检索模式对照。这块的完整踩坑过程见 `develop_docs/学习笔记/RAG评测过程.md`。
- **L3 端到端评测**（`run_eval.py`）——**系统级验收**。真跑 `chat()`，判最终回复的质量 + 工具调用过程。

### 2.2 一条重要的职责分界裁定（面试常被追问）

**指代消解的评测归端到端层，Router 单测只测场景判定。**

为什么？因为"我上次问的那个还有货吗"能不能被正确理解，依赖 `summary` + `products` 商品记忆——那是 **Agent 编排层的职责**，不是 Router 的。Router 只吃 `recent_context`（最近 4 条）和 `summary`（`router.py:41-51`）。如果把指代消解塞进 Router 单测，就会变成用一个**缺了商品记忆的残缺上下文**去考核 Router，测出来的失败是假的。

> 这条裁定的直接后果：曾经存在的 R056/R057 两条"跨压缩指代"Router 用例被**删除**，改由端到端用例 `multiturn_reference_price` 承担。这是一个"发现用例集本身失真"的典型案例——**用例错了比代码错了更危险**。

## 3. L3 端到端评测：从头到尾拆解

这是本项目的评测主战场，我们按"数据 → 采集 → 判分 → 报告"的顺序走一遍。

### 3.1 数据模型：`EvalCase`（`dataset.py:22-53`）

一条用例回答两件事：**输入是什么**、**期望表现是什么**。字段按用途分五组：

| 分组 | 字段 | 说明 |
| --- | --- | --- |
| 标识 | `id` / `category` / `description` | `category` 用于报告分组（主流程 / 对比 / 咨询 / 跨轮指代 / 边界） |
| 输入 | `turns: list[str]` | **多轮**，按顺序喂给**同一个**编排器实例（单轮即 `len==1`） |
| 归因断言 | `expected_route` / `expected_tools` / `tools_success_required` | 路由与工具行为；**归因用**，失败时告诉你锅在哪层 |
| 数值保真断言 | `reply_prices_within` / `must_include_catalog_price(_any)` / `valid_ids_only` / `reply_prices_in_catalog` | 价格与商品 ID 的硬约束（防幻觉核心） |
| 内容断言 | `keyword_hits` / `min_chars` | 政策关键词、最低篇幅 |
| 主观质量 | `judge_aspects` | `["answer_quality", "process"]`；空 = 不判 |

**三个必须理解的设计点**：

1. **留空 = 跳过，不计分**（`dataset.py:12`）。每类用例关心的东西不同：`chitchat_no_tool` 关心"有没有乱调工具"，`detail_price` 关心"价格报得对不对"。如果某条用例没声明 `min_chars`，它就不该因为回复短而被判挂。这个设计避免了"用不相关的断言拖垮聚合"。
2. **`expected_tools` 是 any-of**（`dataset.py:34`）：非空列表 = 至少命中其一；**空列表 = 必须零工具调用**；`None` = 不判。三种语义靠一个字段表达，写在 `check_tools` 里（`metrics.py:118-144`）。
3. **`tools_success_required`**（`dataset.py:38`）——这是被一个真实 bug 逼出来的字段。默认 `True`：要求至少一次**成功**调用（`success=true`）。但"查无结果"类用例（`no_match_honesty`）里，工具**合法地**返回 `success=false`，若按成功口径判，**正确行为必挂**。所以这类用例置 `False`：只要求"发起过调用"。

### 3.2 沙箱隔离：`Sandbox`（`sandbox.py:32-95`）

沙箱做三件事（`sandbox.py:1-18` 的模块 docstring 就是设计说明）：**隔离、插桩、执行**。

#### 隔离（`_build_agent`，`sandbox.py:51-62`）

```python
settings.memory_enabled = True                                  # ← 注意：是 True
settings.memory_dir = str(Path(session_path).with_suffix("")) + "_memory"
settings.mcp_enabled = False
```

三个反直觉点，值得逐个想清楚：

- **为什么 `memory_enabled=True` 而不是关掉？** 因为有一批用例是"记忆驱动"的（`recommend_by_memory`），它们的工具断言要求 LLM 主动调用 `recall_user_memory`。如果把记忆关了，这个工具根本注册不进去，用例**永远失败**——那是被测对象的残缺，不是缺陷。**解法不是关记忆，而是把 `memory_dir` 指向每条用例独立的沙箱目录**：既不读真实用户记忆（可复现），又隔离了用例间状态。
- **为什么 `mcp_enabled=False`？** 两个原因：① 可复现——外网 MCP 服务不稳定；② **幻觉检测需要确定的 ground truth**——只有用本地 mock 目录，`valid_ids_only` 才有意义。
- **每条用例一个独立 session 文件**（`session_path_for`，`sandbox.py:48-49`）：`{tmp_root}/{case_id}.json`。这保证了用例之间零串扰，也支持 `--case-id` 单条重跑。

#### 绝不调 `close()`（`sandbox.py:16`、`sandbox.py:93`）

```python
# 注意：刻意不调用 agent.close()，避免长期记忆巩固写入
```

`MultiAgentOrchestrator.close()`（`orchestrator.py:182-187`）会触发 `consolidate_to_long_term()`——那是一次**真实的 LLM 写入调用**，会：① 污染长期记忆库；② 烧钱；③ 让评测结果不可复现。沙箱只调 `_close_tool_managers()`（`sandbox.py:234-239`）关掉工具连接，不碰记忆。

> 这是"评测环境必须与生产环境在**副作用上**隔离"的经典体现。Agent 的 `close()` 是生产语义的一部分（会话结束→沉淀记忆），但在评测里它是纯副作用。

#### 异常兜底（`sandbox.py:86-88`）

```python
except Exception as e:   # noqa: BLE001
    trace.error = f"{type(e).__name__}: {e}"
```

单条用例崩了不能中断整轮跑批。错误记进 `trace.error`，`CaseResult.passed` 会直接返回 `False`（`evaluator.py:36-39`），但其余用例继续跑。

### 3.3 插桩：零侵入采集的核心（`sandbox.py:98-225`）

**这是整个评测模块最精妙的部分**，也是面试最可能深挖的地方。

#### 为什么能插桩？—— 因为 Agent 内部只有一个 client

`MultiAgentOrchestrator.__init__` 建了**唯一一个** OpenAI client（`orchestrator.py:41-44`），然后把它**注入给所有下游**：

```python
self.client = openai.OpenAI(api_key=..., base_url=...)   # orchestrator.py:41
self.router = Router(self.client, self.model)            # orchestrator.py:57
# ↑ 所有 SubAgent 也共用这一个 client（orchestrator.py:61-78）
```

所以——**给这一个 client 的两个方法打补丁，就能捕获整个会话（Router + 所有子 Agent + Result）的全部 LLM 调用**。不需要改 `orchestrator.py`、不需要改 `agents.py`、不需要改 `router.py`。这就是"零侵入"的物理基础。

#### 四个补丁点（`_instrument`，`sandbox.py:98-123`）

| # | 补丁对象 | 方法 | 采集什么 |
| --- | --- | --- | --- |
| 1 | `agent.client.chat.completions` | `create` | 所有普通生成调用：token / 延迟 / 请求的工具 / `finish_reason` |
| 2 | `agent.client.beta.chat.completions` | `parse` | 结构化输出调用（当前主要归 `extract`） |
| 3 | 每个 `ToolManager` | `execute_tool` | 工具名 + 入参 + 返回信封（**数值保真的 ground truth 来源**） |
| 4 | `agent.router` | `route` | 每轮路由结果（放到 `trace._pending_route`） |

打补丁的写法是"保存原方法 → 换成 wrapper → 记进 `patches` 列表"，`finally` 里还原（`sandbox.py:88-90`）：

```python
patches.append((completions, "create", completions.create))   # 存原件
completions.create = self._wrap_create(completions.create, trace)  # 换包装
...
finally:
    for obj, attr, original in patches:
        setattr(obj, attr, original)   # 无论成败都还原
```

> **为什么一定要还原？** 因为这个 client 是模块级/实例级对象，`wrapt` 之类的全局补丁会跨用例泄漏。`finally` 还原保证了"用例 A 的插桩不会污染用例 B"。注意 `_wrap_execute_tool` 是**实例属性覆盖**（`tm.execute_tool = ...`），只影响这一个 ToolManager 实例。

#### 工具的插桩：同时喂两个下游（`sandbox.py:150-162`）

```python
def wrapper(name, arguments):
    with self.reporter.tool_span(name, dict(arguments)) as span:   # → Langfuse（§5.3）
        result_str = original(name, arguments)
        if span is not None:
            span.update(output=result_str)
    trace.tool_observations.append(                                    # → RunTrace（§3.4）
        ToolObservation(name=name, arguments=dict(arguments), result=result_str))
    return result_str
```

一个包装，两个消费者：Langfuse 看板（人看）与 `RunTrace`（机器判分）。

#### `purpose` 标注：一个被预算变更打破两次的启发式（`sandbox.py:177-192`）

`_guess_purpose` 给每次 LLM 调用打上 `router` / `react` / `answer` 标签，用于报告可读性。

**它的判据改过两次，这个演进过程本身就是很好的工程教训**：

| 版本 | 判据 | 为什么失效 |
| --- | --- | --- |
| v1 | `max_tokens == 10` → router | Router 预算改到 512，"10"过时了 |
| v2 | `max_tokens <= 1024` → router | Router 预算 512→2048，router 被误标成 answer |
| **v3（现行）** | **调用位置驱动**：在 `Router.route()` 作用域内 → router；带 `tools` → react；其余 → answer | 不依赖预算数值，抗变更 |

v3 的实现靠一个标志位：`_wrap_route` 进入时置 `self._in_router = True`，`finally` 复位（`sandbox.py:164-174`）。

> **教训**：不要用"魔法数值特征"去识别语义。数值是别人随手会改的配置；**调用位置**才是稳定的事实。

### 3.4 `RunTrace`：唯一数据载体（`trace.py`）

`RunTrace`（`trace.py:37-52`）是"采集"与"评分"之间的**契约**。判分器只读它，不碰 Agent 内部。

| 字段 | 含义 |
| --- | --- |
| `turns` / `replies` | 每轮输入 / 每轮最终回复（**对齐**） |
| `routes` | 每轮路由结果（与 turns 对齐） |
| `llm_calls: list[LLMCallRecord]` | 每次 LLM 调用的 token / 延迟 / 请求的工具 / `finish_reason` |
| `tool_observations: list[ToolObservation]` | 每次工具调用的入参 + 返回信封 |
| `tool_boundaries: list[int]` | **轮次边界**：第 i 项 = 第 i 轮结束时的观测数 |
| `langfuse_trace_id` | 上报开启时该用例的 trace id（判分后挂 score 用） |
| `error` | 运行异常；`None` = 正常 |

#### `abnormal_llm_calls`：截断诊断入口（`trace.py:72-83`）

```python
return [c for c in self.llm_calls
        if c.finish_reason and c.finish_reason not in ("stop", "tool_calls")]
```

**判据的细化很关键**（注释里写了 2026-09-16 实测）：

- `finish_reason == "stop"` → 正常结束
- `finish_reason == "tool_calls"` → **也是正常的**！这是 ReAct 循环请求工具的表现（该轮跑批 46/128 次），**不等于异常**。如果不细看，会误以为一半调用都异常了。
- `finish_reason == "length"` → **异常**：输出被截断（回复中途断掉，或被"思考"吃光了预算）
- 空字符串 → 未采集到，**不判**异常

> 这个"`tool_calls` 不是异常"的修正，是评测仪表**自身正确性**的修正——**错误的仪表比没有仪表更糟**，它会把你引向错误的优化方向。

#### `tool_boundaries` 为什么存在（`trace.py:47-50`）

为了修 `judges.py` v1 的一个真实 bug（见 §3.6）：multi-turn 用例里，judge 材料如果把"最后一轮的问题"配"全会话的工具调用序列"，那么第一轮的**正当工作**会被判成"与当前问题无关"。有了边界，就能切成 `tool_observations[bounds[i-1]:bounds[i]]` 精确对齐每轮。

### 3.5 规则判分：9 条确定性断言（`metrics.py`）

`_CHECKS`（`metrics.py:273-283`）就是全部断言，按顺序执行：

| # | 断言 | 检查什么 | 关键设计 |
| --- | --- | --- | --- |
| 1 | `check_route` | 每轮路由 == 期望 | 支持 `"presale\|consult"` 二选一（模糊输入合理路由不止一种） |
| 2 | `check_tools` | 工具行为符合期望 | any-of；空列表=零调用；`tools_success_required` 切换口径 |
| 3 | `check_valid_ids` | 回复中商品 ID 都真实存在 | **防幻觉核心** |
| 4 | `check_prices_within` | 价格 ∈ 指定品类目录且 ≤ 上限 | 支持 `last_turn_only`（中途改预算场景） |
| 5 | `check_must_include` | 每个 (ID, 目录价) 对都出现 | 精确价断言（全部） |
| 6 | `check_must_include_any` | 命中任意一对即可 | 多轮指代场景用 |
| 7 | `check_prices_in_catalog` | 所有价格都真实存在 | 防编造价格 |
| 8 | `check_keywords` | 最终回复命中关键词 | 检查**最后一轮** |
| 9 | `check_min_chars` | 最终回复最低篇幅 | 检查**最后一轮** |

#### ground truth 从哪来

`metrics.py:41-50`：从 `app/db/snapshot.py` 的 `PRODUCTS` 建索引——

```python
_PRODUCT_BY_ID = dict(PRODUCTS)                  # {product_id: 商品}
_CATEGORY_PRICES = {category: {price, ...}}      # 品类 → 合法价格集合
_ID_PATTERN = re.compile(r"\b(?:LP|PH|HP)-\d{2}\b")
_PRICE_PATTERN = re.compile(r"[¥￥]\s*(\d{3,6})|(\d{3,6})\s*元")
```

`PRODUCTS` 是 PostgreSQL 真值层的快照。**这就是"确定性"的来源**——判分不查库、不联网、不调 LLM，只比对内存中的静态目录。

#### 五个"防冤"细节（这些才是断言写得好不好的分水岭）

1. **价格必须带货币标记**（`metrics.py:49-50`）：`¥5180` 或 `5180 元`。为什么？否则"16GB""14 寸"这类规格数字会被误当价格。
2. **千分位归一化**（`_normalize`，`metrics.py:53-55`）：LLM 常写 `¥5,180`，目录里是 `5180`。不归一化就会误判。
3. **建议性数字剔除**（`check_prices_in_catalog`，`metrics.py:236-237`）：`¥700-1000`（区间）、`¥1000 左右`（约数）是**给用户的预算建议**，不是商品报价，不算编造。
4. **用户回显不算编造**（`metrics.py:240`）：用户自己输入的预算数字被 Agent 复述，不算编造价格。
5. **失败调用不算命中**（`successful_tool_names`，`metrics.py:97-115`）：结果信封 `success=false`（如查询无结果）不算"信息需求被满足"。这条注释里写了教训来源：`recommend_reviews` 曾出现两次 0 结果的错误调用却命中了 `expected_tools`。

#### `CheckResult` 的三态（`metrics.py:23-38`）

```python
CheckResult(name, applicable, passed, detail)
```

- `applicable=False` → 用例没声明这个断言，**不计分**（`CheckResult.skip`）
- `applicable=True, passed=False` → 判挂，`detail` 说明原因
- `detail` 一律写成**人话解释**（如"提及的目录价最低 1299，仍超预算 1500"），因为报告要直接给人看

#### 判分器自己也要被测：`--self-test`（`run_eval.py:43-71`）

```
构造一条"编造了 LP-99（不存在的 ID）+ ¥9999（不存在的价格）"的假轨迹
→ 断言 valid_ids 与 prices_in_catalog 必须判挂
→ 若漏检，打印 "判分器漏检！" 并退出码 1
```

**为什么需要它**：如果判分器本身有 bug（比如正则写错、阈值写反），它会**静默地**把所有用例都判成通过——你会以为系统 100% 健康。**评测仪的校准和被测系统的测试同等重要。**

### 3.6 LLM judge：主观维度的 advisory 评分（`judges.py`）

规则断言能查"价格对不对""ID 假不假"，但查不了"回答结构清不清晰""工具调用有没有冗余"。这部分交给 LLM judge。

#### 两个维度 + 版本化 rubric（`judges.py:38-67`）

| aspect | 评什么 | 材料 |
| --- | --- | --- |
| `answer_quality` | 最终回答的质量（1-5 分） | 各轮用户问题 + **最终回复** |
| `process` | 工具调用过程的必要性与效率（1-5 分） | **按轮对齐**的"该轮问题 → 该轮调用序列" |

`JUDGE_PROMPT_VERSION = "v2"`（`judges.py:32`），每次落盘都带上版本号——**分数必须能归因到"哪一版的评分标准"**，否则跨版本比较毫无意义。

#### 防抖动三件套（`judges.py:6-9`）

| 手段 | 实现 | 解决什么 |
| --- | --- | --- |
| `temperature=0` + 固定 rubric 版本 | `judges.py:33` | 同回复同分数，才可归因 |
| 强制 JSON 输出 + 解析校验 | `_parse_judge_response`（`judges.py:70-88`） | 输出可用；容忍 `4.0` 这类整数型浮点，拒绝越界值 |
| 失败重试一次 → 再失败返回 `None` | `judge_aspect`（`judges.py:171-181`） | **宁可不判，不打假分**（报告标"未判"） |

#### v2 修复的两个真实 bug

1. **材料错位**（`judges.py:11-15`）：v1 只给最后一轮问题却给全会话调用序列，导致 multi-turn 用例第一轮的正当工作被判"与当前问题无关"（实测 `multiturn_switch_consult` 冤枉 1/5）。修法：材料按轮组织，用 `tool_boundaries` 切分。
2. **长 reasons 截断**（`judges.py:17`）：`max_tokens` 1024→2048，且 reasons 限 2 条。修法见 `_judge_once`（`judges.py:121`）。

#### 一个耐人寻味的 rubric 补充（`judges.py:61-63`）

```
设计说明：search_catalog（确定性过滤）与 search_products（语义检索）
双路并调是设计内的合规校验模式，不计为冗余；
但同一工具重复调用且参数高度重叠仍计冗余。
```

**为什么要把"设计意图"写进评分标准？** 因为 judge 只看材料，不知道架构。如果不告诉它"双路并调是我们故意的交叉校验"，它会把正确的架构判成冗余调用——**这是"用架构知识校准评测标准"的典型例子**。反过来说，protect 部分要收窄：同工具同参数仍算冗余，否则就把真问题也豁免了。

#### 失败落盘（`judges.py:91-108`）

解析最终失败时把原始输出写进 `app/evaluation/runs/judge_failures.jsonl`，并**记录 `finish_reason`**：

- `length` → 输出被截断 → **该加预算**
- `stop` → 格式不合规 → **该改 prompt**

**两类问题的修法完全不同，所以必须区分记录**。"未判"必须留下可查证据，否则你只知道"有 2 条没判分"，却不知道该修哪。

#### 为什么 judge 不参与 `passed`

`evaluator.py:30`：`judge_scores` 只写进 `CaseResult.judge_scores`，`passed` 只看 `checks`（规则断言）。

理由：**通过门槛必须是确定性的**。judge 有温度、有格式抖动、有误判——它可以作为"过程洞察"（发现口碑检索冗余调用就是它发现的），但不能作为"回归红线"。`settings.eval_pass_threshold`（`settings.py:69`）是留给"judge 升格为门槛"的第三步用的，当前未启用。

### 3.7 执行器与报告（`evaluator.py` / `run_eval.py`）

#### 编排顺序（`Evaluator.run_case`，`evaluator.py:111-138`）

```
sandbox.run(case) → RunTrace
  ├─ 有 error？→ 记 CaseResult(error=...)，跳过判分
  └─ 无 error
       ├─ metrics.run_checks(case, trace)      # 9 条规则断言
       ├─ self._run_judges(case, trace)        # LLM judge（若声明了 judge_aspects）
       └─ 计算 tool_hits                        # 期望工具命中数 / 总数
  └─ reporter.report_case(...)                  # 把判分结果挂回 Langfuse trace
```

`passed` 的定义（`evaluator.py:36-39`）：`error` 为空 **且** 所有 `applicable` 的 check 都 `passed`。

#### `_judge_materials`：按轮组装（`evaluator.py:163-188`）

```python
bounds = trace.tool_boundaries
aligned = len(bounds) == len(trace.turns)     # 边界完整才按轮切
...
obs = trace.tool_observations[start:bounds[i]] if aligned else ...
```

**兜底逻辑值得注意**：如果边界不完整（异常中途退出），**把全序列挂到最后一轮**——"宁可材料粗，不至错配"。这是一个"降级但不给错"的设计取向。

#### 终端报告三层（`_print_overview` / `_print_report`，`run_eval.py:83-198`）

1. **概览矩阵**——`用例 × 分层`：

```
  用例                          结果 路由 工具      结果断言  质量  过程   token
  detail_price                ✅  ✓    ✓ 1/2     ✓ 2项    4/5   3/5    15,432
```

> 列结构直接对应三层断言：**哪列先出 ✗，问题就在哪层**。这是"可归因"约束在报告层的落地。

2. **汇总**——通过率 / token 分布 / judge 均分 / 断言全貌 / 截断诊断
3. **逐条明细**——每条用例的每个 check 的 `detail` + judge 的理由

#### 落盘与清洁（`run_eval.py:258-285`）

- 每条用例：`runs/<case_id>.json`（`RunTrace.to_dict()`）
- 聚合报告：`runs/report.json`
- **清理陈旧轨迹**：全量跑批时删除不属于本轮用例集的历史 json。注释写了教训：“否则后续分析会把历史文件当成本轮结果（2026-09-16 误判过一次）”。

> 这条看似琐碎的清理逻辑，解决的是一个真实踩过的坑：**产物目录不做生命周期管理，早晚会有人（包括 AI）拿过期文件当最新数据**。

## 4. L1 Router 单元评测（`run_router_eval.py`）

L1 与 L3 是**两种方法论**，对比着看最能理解分层意义。

| 维度 | L1 Router | L3 端到端 |
| --- | --- | --- |
| 是否建 Agent | ❌ 只建 `Router` | ✅ 完整 `MultiAgentOrchestrator` |
| 是否用沙箱 | ❌ 直接调 | ✅ 隔离 + 插桩 |
| 判定 | `actual == expected`（**完全一致**） | 9 条规则断言 |
| 单条成本 | 1 次 LLM 调用 | 十几次 |
| 输出 | 准确率 + 场景级 P/R/F1 + 分维度/分难度 | 通过率 + 断言明细 |

### 4.1 用例结构（`router_dataset.py:19-30`）

```python
RouterCase(id, group, difficulty, description, input, expected, history, summary)
```

- `expected` 是**有序**场景列表，要求完全匹配（`SCENARIO_ORDER` 固定顺序）。
- `group` 是评测维度：`single_presale / single_consult / multi / continue / switch / boundary`。
- `difficulty`：easy / medium / hard。
- `summary` 可选：模拟"历史已被压缩"场景，验证跨压缩指代兜底。

### 4.2 多维指标（`run_router_eval.py:76-111`）

不只看总准确率，还看**场景级多标签**的精确率 / 召回率 / F1（因为一条 `multi` 用例的期望有多个场景），以及**分维度、分难度**的通过率。这样"准确率掉了 2%"能立刻定位成"掉在 `switch` 维度的 hard 用例上"。

### 4.3 CI 门禁（`run_router_eval.py:127-137`）

```python
if args.min_accuracy is not None and accuracy < args.min_accuracy:
    sys.exit(1)
if n_ok != n:
    sys.exit(1)      # 当前基线 100%，任何回归立刻让 CI 变红
```

当前基线 **55/55 = 100%**（`app/evaluation/runs/router_report.md`，2026-09-08）。因为达到了 100%，这个退出码成了一根很紧的红线——**改 `ROUTER_PROMPT` 必须重跑**。

### 4.4 顺带一提：Router 的截断根因

`router.py:53-70` 有一长段注释记录了根因定位：

```
现象：路由用例长期不稳，completion=512、finish_reason=length
根因：路由未传 reasoning_effort → 走服务端默认（偏高的）思考强度
     → 推理模型思考与正文共享 max_tokens → 正文被挤空/截断
修法：reasoning_effort="low" + max_tokens=2048（保险）
```

**这个 bug 是被评测发现的**——`abnormal_llm_calls` 把 `finish_reason=length` 暴露出来，才有得查。这就是"可观测性 → 可归因 → 可修复"的完整闭环，也是评测体系价值的最好证明。

## 5. Langfuse 平台化（重点章节）

前面全是"本地能跑通"的部分。Langfuse 解决的是**另一个维度的问题**：跑批数据散落在几十个 json 文件里，人看不过来，也没法跨跑批对比。这一节从头讲清它的概念、接入方式、以及"我们到底加了什么"。

### 5.1 Langfuse 是什么：五个核心概念

Langfuse 是一个**LLM 应用可观测性平台**。它的数据模型是树状的，你只需要记住 5 个词：

| 概念 | 含义 | 在本项目里的映射 |
| --- | --- | --- |
| **Trace** | 一次完整请求的运行记录（树根） | **一条评测用例**（`eval:<case_id>`） |
| **Observation** | trace 内的一个节点（span / generation） | LLM 调用、工具调用、judge 调用 |
| **Session** | 一组 trace 的归组（用户会话级） | **一次跑批**（`run_id = YYYYMMDD-HHMMSS`） |
| **Score** | 挂在 trace 上的评分 | `e2e_pass` / `judge_*` / `tools_hit_rate` |
| **Generation** | 特化的 observation，带 model / token / prompt 信息 | 由 drop-in 自动生成的 LLM 节点 |

**本地 json vs 平台看板的区别**：

| | 本地 `runs/*.json` | Langfuse 看板 |
| --- | --- | --- |
| 看一条用例 | ✅ 可以 | ✅ 可以（可视化树） |
| 跨跑批对比（这次 vs 上次） | ❌ 得自己写脚本 diff | ✅ Sessions / Scores 页直接排 |
| 按分数排序找最差的用例 | ❌ | ✅ |
| 定位"哪次 LLM 调用吃了 8000 token" | 翻 json | ✅ 树上一眼看到 |
| 分享给他人 | 发文件 | ✅ 发链接 |

一句话：**本地 json 是"证据留档"，Langfuse 是"数据分析与协作"**。两者互补，所以我们的设计是"本地为主、平台为旁路"。

### 5.2 它是怎么接进来的：两条独立的路径

理解 Langfuse 接入，关键是分清**两条路径**——它们来源不同、机制不同、风险也不同。

#### 路径 A：drop-in 自动捕获（一行 import 的事）

看 `orchestrator.py` 的第 16 行：

```python
#from openai import OpenAI          ← 原来的写法（已注释）
from langfuse.openai import openai  ← 现在的写法
```

**就这一行**，Langfuse 就接进来了一半。机制是 **wrapt 类级全局补丁**：

- `langfuse.openai` 模块在 import 时，会给 OpenAI SDK 的**类方法**打补丁（不是某个实例）；
- 因为是**类级**补丁，进程内**所有** `OpenAI` 实例的方法都被包装了；
- 所以 `orchestrator.py:41` 建的 client 自动被捕获，不需要任何额外代码。

**这一行 import 带来的链式效果**（`app/multi_agent/__init__.py:9-11`）：

```
import app.multi_agent
  └─ from app.multi_agent.orchestrator import MultiAgentOrchestrator
       └─ from langfuse.openai import openai      ← 补丁在此刻生效（进程级）
```

所以**任何** `import app.multi_agent` 的进程，它的所有 OpenAI 调用都会被记录（前提是配了 KEY）。这解释了 `settings.py:103-107` 那句注释："只要进程 import 了 `app.multi_agent` …… 所有 OpenAI 调用都会被接入"。

**drop-in 的副作用**（§5.4 坑 3 的来源）：judge 用的 `client`（`run_eval.py:248` 建的那个）**也被同一个类级补丁接住了**。而 judge 是在 `case_scope` 之外调用的——不处理的话，每次 judge 都会变成一个**独立的根 trace**，看板上瞬间多出 20 条用例 × 2 个 aspect 的噪音。

#### 路径 B：显式上报（`reporter.py` 的四个接口）

drop-in 只能自动捕获 LLM 调用。下面这些东西它捕获不到，必须显式上报：

| 要上报的东西 | 接口 | 代码 |
| --- | --- | --- |
| 用例作用域（trace 根 + session 归组） | `case_scope()` | `reporter.py:58-77` |
| 工具调用（input/output） | `tool_span()` | `reporter.py:79-88` |
| judge 调用（挂回用例 trace） | `judge_scope()` | `reporter.py:90-106` |
| 判分结果 | `report_case()` | `reporter.py:119-151` |

**两条路径的分工**：A 负责"LLM 调用"（drop-in 最擅长），B 负责"业务语义"（trace 边界、工具、分数）。它们靠 **OTel 上下文传播**拼到同一棵树里——显式上报开了 `case_scope` 后，drop-in 捕获的 generation 会自动挂到"当前 span"下（`reporter.py:14-15` 注释："实测同 trace，无需注入 trace_id"）。

### 5.3 四个上报接口逐个拆

#### `case_scope`：一条用例的根（`reporter.py:58-77`）

```python
with propagate_attributes(
    session_id=self.run_id,          # 一次跑批 = 一个 session
    tags=[case_id, category],        # 标签：可按用例名/分类筛
    trace_name=case_id,              # 看板上 trace 显示成用例名，而不是随机 id
):
    with self._client.start_as_current_observation(
        name=f"eval:{case_id}", as_type="span",
        input={"case_id": case_id, "category": category},
    ) as span:
        yield span.trace_id          # ← yield 出 trace_id，供后续挂 score
```

三个设计点：

1. `propagate_attributes` 设置 **trace 级**属性（session / tags / trace_name）。注意注释 `reporter.py:11-12` 特别提醒：**trace 归组与 session 走 OTel 上下文传播，不通过 `create()` 的 kwarg**——因为那些 kwarg 会漏给真 API 报错。
2. `start_as_current_observation(as_type="span")` 建立"当前 span"，之后 drop-in 的 generation 与我们的工具 span **自动挂到它下面**。
3. **yield `trace_id`** 是必要的：判分发生在沙箱跑完之后，那时作用域已退出，只剩一个 trace_id 可以用来 `create_score`。

#### `tool_span`：单次工具调用（`reporter.py:79-88`）

```python
with self._client.start_as_current_observation(
    name=f"tool.{name}", as_type="span", input=arguments,
) as span:
    yield span
```

调用方（`sandbox.py:154-157`）拿到 span 后 `span.update(output=result_str)`。**语义**：input = 工具入参，output = 工具返回信封。这样看板上"哪个工具被调了几次、参数是什么、返回了什么"一清二楚——排查冗余调用就靠这个。

#### `judge_scope`：把 judge 挂回用例 trace（`reporter.py:90-106`）

```python
with self._client.start_as_current_observation(
    name=f"judge:{aspect}", as_type="span",
    trace_context={"trace_id": trace_id},   # ← remote parent：显式指定父 trace
) as span:
    yield span
```

**这是整章最值得理解的一处设计**。为什么需要它？

- judge 调用发生在 `CaseResult` 已经拿到 trace 之后（`evaluator.py:153`）；
- 此时 `case_scope` 已退出，**没有"当前 span"**；
- 但 judge 的 client 因 wrapt 类级补丁**仍是 instrumented 的**；
- 结果：judge 调用没有父节点 → 每次生成一个**独立的根 trace** → 看板噪音。

`trace_context={"trace_id": ...}` 就是告诉 Langfuse："这个 span 的父节点不在当前上下文里，请挂到这个 trace 上"。挂回去之后，**判分过程与 Agent 轨迹在同一棵树里**——可审计、无污染。

#### `report_case`：判分结果 → Score（`reporter.py:119-151`）

```python
self._client.create_score(trace_id=..., name="e2e_pass",
                          value=passed, data_type="BOOLEAN", comment=...)
for aspect, judged in judge_scores.items():
    self._client.create_score(..., name=f"judge_{aspect}",
                              value=judged["score"], data_type="NUMERIC", ...)
if tool_hits and tool_hits[1]:
    self._client.create_score(..., name="tools_hit_rate",
                              value=tool_hits[0] / tool_hits[1], data_type="NUMERIC")
```

三类 score，`comment` 写成人话（"断言失败: valid_ids、prices_in_catalog" / "全部断言通过"）。**关键设计**：`create_score` 在判分**之后**调用，且整段包在 `try/except` 里，失败只打 warning（`reporter.py:150-151`）——**上报是旁路，绝不影响判分结果**。

### 5.4 三个"坑"与对应设计（理解接入的关键）

| 坑 | 现象 | 根因 | 我们的修法 | 代码 |
| --- | --- | --- | --- | --- |
| **① 凭证从哪来** | 某些入口（router/rag 评测、build_* 脚本）静默不上报，每次调用打一条告警 | SDK **只从 `os.environ` 读**凭证；而 pydantic 的 `env_file` 只把 `.env` 读进 **settings 对象**，不进环境变量 | **凭证桥接**：`settings.py` 末尾把 settings 里的三个 langfuse 字段 `setdefault` 进 `os.environ`。任何入口只要 `import settings` 即生效 | `settings.py:102-117` |
| **② `name=` 不能乱传** | 传了会在真 API 报"未知参数" | `name` 是 Langfuse 专用 kwarg；如果 client **没被** drop-in 包装（如没装 langfuse），它会被原样透传给 OpenAI | 先检查"是否已被包装"（`is_instrumented` 看方法类型是不是 `BoundFunctionWrapper`），确认后才注入 `name` | `reporter.py:108-117`、`sandbox.py:130-131` |
| **③ judge 噪音** | 看板上突然多出大量独立根 trace | judge 的 client 也被 wrapt 类级补丁接住，但调用时没有"当前 span" | `judge_scope` 用 `trace_context` 显式挂回用例 trace | `reporter.py:90-106` |

**坑 ① 的代码**（`settings.py:111-117`）：

```python
for _key, _value in (
    ("LANGFUSE_BASE_URL", settings.langfuse_base_url),
    ("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key),
    ("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key),
):
    if _value:
        os.environ.setdefault(_key, _value)   # setdefault：shell 显式变量优先
```

注意 `setdefault` 而非直接赋值——**shell 里显式 `export` 的变量优先级更高**，方便临时切环境。另外注释提到一个细节：SDK 同时认 `LANGFUSE_BASE_URL` 与 `LANGFUSE_HOST` 两个名字，项目 `.env` 用的是前者。

**坑 ② 的检查逻辑**（`reporter.py:114-115`）：

```python
return type(client.chat.completions.create).__name__ == "BoundFunctionWrapper"
```

`BoundFunctionWrapper` 是 wrapt 包装后的类型名。**这个检查本身就是对"补丁是否生效"的运行时探测**——比"检查 langfuse 是否 import 了"可靠得多。

### 5.5 降级设计：全链路 no-op

这是整个上报层最重要的**健壮性**设计。原则：**上报是旁路，任何失败都不能影响本地评测**。

```
LangfuseReporter(run_id, enabled=True)
  └─ _build_client()
       ├─ 没配 LANGFUSE_PUBLIC_KEY / SECRET_KEY  → return None（打 info 日志）
       ├─ 没装 langfuse 依赖 / 构造异常          → return None（打 warning）
       └─ 成功                                    → self.enabled = True
```

`enabled=False` 时，**每一个接口都是 no-op**：

| 接口 | no-op 行为 |
| --- | --- |
| `case_scope` | `yield None`（`trace_id=None`） |
| `tool_span` | `yield None` → 调用方 `if span is not None` 跳过 update |
| `judge_scope` | `yield None` |
| `report_case` | 直接 return（`if not (self.enabled and trace_id)`） |
| `flush` | 直接 return |

还有一层开关：`run_eval.py:239` 的 `enabled=not args.no_report`——**`--no-report` 可以强制关掉**，离线或不想产生云端数据时用。

**两条数据流图**：

```
有 KEY：  用例 → Sandbox(reporter) → case_scope 开 trace
                 ├─ LLM 调用 → drop-in 自动 generation → 挂当前 span
                 ├─ 工具调用 → tool_span 显式 span
                 └─ judge   → judge_scope 挂回 trace
            → report_case → create_score × N
            → flush() → 云端

无 KEY：  用例 → Sandbox(disabled) → case_scope yield None
                 ├─ LLM 调用 → 无包装，原样发 HTTP
                 ├─ 工具调用 → tool_span yield None，纯旁路
                 └─ judge   → judge_scope yield None
            → report_case 直接 return
            → flush() 直接 return
          （本地 runs/*.json 与终端报告完全不受影响）
```

### 5.6 你在看板上会看到什么

配置好 KEY 跑一次 `run_eval.py` 后：

| 看板位置 | 看到什么 |
| --- | --- |
| **Sessions 页** | 一次跑批（`20260916-185811` 这种 run_id），点进去是本次全部 20 条用例 |
| **Traces 列表** | 20 条 trace，名字就是 `case_id`（如 `detail_price`），可按 tag 筛（`case_id` / `category`） |
| **单条 trace 树** | `eval:<case_id>` (span) ├─ `agent:router` (generation) ├─ `agent:react` ├─ `agent:answer` ├─ `tool.search_products` (span) ×N ├─ `judge:answer_quality` (span) └─ `judge:process` |
| **Scores** | `e2e_pass`（布尔）、`judge_answer_quality`（1-5）、`judge_process`（1-5）、`tools_hit_rate`（0-1） |

**这正是 `name=` 注入的用途**（`sandbox.py:131`）：不注入的话，看板上所有 LLM 节点都叫 "OpenAI-generation"，根本分不清哪个是路由、哪个是回答。注入了才显示 `agent:router` / `agent:react` / `agent:answer`（`_guess_purpose` 的产出，§3.3）。

## 6. 一次跑批的完整时序

把前面所有零件串起来。跑 `python app/scripts/run_eval.py` 时发生的事：

| # | 步骤 | 代码位置 |
| --- | --- | --- |
| 1 | `import app.evaluation...` 触发 `import app.multi_agent` → **langfuse drop-in 补丁生效（进程级）** | `orchestrator.py:16` |
| 2 | `import app.config.settings` → **凭证桥接进 `os.environ`** | `settings.py:102-117` |
| 3 | 建 `run_id`（= session id）+ `LangfuseReporter` | `run_eval.py:238-242` |
| 4 | 建 judge 用的 `OpenAI` client（**也被 drop-in 接住**） | `run_eval.py:248` |
| 5 | 逐条用例：`Sandbox.run(case)` | `sandbox.py:64-95` |
| 6 | └ 开 `case_scope` → 建 `eval:<case_id>` span，拿到 `trace_id` | `reporter.py:58-77` |
| 7 | └ 建 Agent（隔离 settings），打 4 类补丁 | `sandbox.py:51-62`、`98-123` |
| 8 | └ 逐轮 `agent.chat(turn)`；每轮收割 `_pending_route`、记 `tool_boundaries` | `sandbox.py:78-84` |
| 9 | │ └ Router LLM 调用 → `_wrap_create`（purpose=router，带 name）→ drop-in 上报 | `sandbox.py:125-137` |
| 10 | │ └ 子 Agent ReAct → `_wrap_create`（purpose=react）+ `_wrap_execute_tool` | `sandbox.py:150-162` |
| 11 | └ `finally`：还原全部补丁、关工具连接（**不调 close()**） | `sandbox.py:88-93` |
| 12 | `metrics.run_checks(case, trace)` → 9 条断言 | `evaluator.py:121` |
| 13 | `_run_judges` → 每个 aspect 开 `judge_scope` 挂回 trace → `judge_aspect` | `evaluator.py:140-161` |
| 14 | `report_case` → `create_score` × N | `reporter.py:119-151` |
| 15 | 聚合 → 终端三层报告 | `run_eval.py:83-198` |
| 16 | 落盘 `runs/<case_id>.json` + `report.json`（清理陈旧轨迹） | `run_eval.py:258-285` |
| 17 | `flush()` → 云端落地；打印示例 trace 链接 | `run_eval.py:287-293` |

## 7. 怎么跑、怎么看

> 按你的约定：**长跑批你自己执行**，我只给命令。以下命令供你直接复制。

| 目的 | 命令 | 说明 |
| --- | --- | --- |
| 判分器自检（不调 LLM，秒级） | `python app/scripts/run_eval.py --self-test` | 验证编造数据能被查出来 |
| 单条端到端调试 | `python app/scripts/run_eval.py --case-id detail_price` | 只跑一条，看报告明细 |
| 全量端到端 | `python app/scripts/run_eval.py` | 20 条，成本约几毛 |
| 端到端（关上报） | `python app/scripts/run_eval.py --no-report` | 离线/不产生云端数据 |
| Router 评测 | `python app/scripts/run_router_eval.py` | 55 条 |
| Router 评测 + CI 门禁 | `python app/scripts/run_router_eval.py --min-accuracy 0.98` | 低于门槛退出码 1 |
| RAG 评测 | `python app/scripts/run_rag_eval.py` | 152 条 |
| 评测模块单测 | `pytest tests/test_evaluation.py -v` | 38 个用例，不调 LLM |

**产物在哪**：

| 产物 | 路径 |
| --- | --- |
| 每条用例轨迹 | `app/evaluation/runs/<case_id>.json` |
| 聚合报告 | `app/evaluation/runs/report.json` |
| Router 报告 | `app/evaluation/runs/router_report.md` |
| judge 未判证据 | `app/evaluation/runs/judge_failures.jsonl` |
| RAG 报告 | `report/` 与 `app/evaluation/runs/`（见 §8 边界说明） |

**当前基线（2026-09-16）**：L1 Router 55/55 = 100%；L3 端到端 20/20，68/68 断言通过，无截断告警。

## 8. 已知边界与坑（诚实的局限清单）

读完上面所有"设计得多好"的内容，这一节是**反面清单**。理解局限比理解优点更能体现工程成熟度。

| # | 局限 | 具体是什么 | 影响 / 待办 |
| --- | --- | --- | --- |
| 1 | **L3 证据强度弱于其他两层** | `runs/rag_report.json` 实际不存在，RAG 基线的部分数字来自早期报告 | 面试表述需注明口径；建议重跑刷新 |
| 2 | **judge 只是 advisory** | 不能作为回归红线；有温度与格式抖动 | 设计如此，非缺陷；升格为门槛是"第三步" |
| 3 | **20 条用例的代表性** | 用例集规模小，且是人工设计——存在"用例是否失真"的风险 | 曾发生 R056/R057 这类失真用例（已删）；需持续审视 |
| 4 | **沙箱 ≠ 生产** | 关了 MCP、mock 工具、独立 memory_dir | 无法覆盖 MCP 集成路径；这是"可复现"的代价 |
| 5 | **规则断言的能力边界** | 只能查可形式化的东西（价格/ID/关键词/篇幅） | "回答有没有帮助"这类判断必须靠人看或 judge |
| 6 | **`_guess_purpose` 仍是启发式** | 位置驱动已比数值驱动稳，但"answer"是兜底桶 | 报告可读性用途，不作硬断言，风险可控 |
| 7 | **压缩节省率 / 并行加速比未测** | 见 `导学-PickWise.md §9` | 待测 |
| 8 | **Router 单测不管指代消解** | 职责分界裁定（§2.2） | 指代消解由 L3 承担，勿再加回 Router 用例 |

## 9. 读完能回答什么（自测题）

| # | 问题 | 答案位置 |
| --- | --- | --- |
| 1 | 插桩为什么不用改产品代码？物理前提是什么？ | §3.3 / `orchestrator.py:41,57` |
| 2 | 沙箱为什么 `memory_enabled=True` 却要改 `memory_dir`？ | §3.2 |
| 3 | 沙箱为什么绝不调 `agent.close()`？ | §3.2 / `sandbox.py:16,93` |
| 4 | `finish_reason=tool_calls` 算异常吗？ | §3.4 / `trace.py:72-83`（不算） |
| 5 | `details` 为空的用例会被判挂吗？ | §3.1（不会，留空=跳过） |
| 6 | `tools_success_required` 解决什么？ | §3.1（查无结果类用例的"防冤"） |
| 7 | `tool_boundaries` 是为了修什么 bug？ | §3.4 / §3.6（judge 材料错位） |
| 8 | judge 为什么不参与 pass/fail？ | §3.6 |
| 9 | Langfuse 的 trace / session / observation / score 分别对应什么？ | §5.1 |
| 10 | drop-in 和显式上报各负责什么？怎么拼到同一棵树？ | §5.2 |
| 11 | `judge_scope` 为什么必须显式指定 `trace_context`？ | §5.3 / §5.4 坑 3 |
| 12 | Langfuse 凭证为什么要在 settings 里桥接进 `os.environ`？ | §5.4 坑 1 |
| 13 | `name=` 参数为什么不能无脑传？ | §5.4 坑 2 |
| 14 | 没配 KEY 时，本地评测会受影响吗？ | §5.5（不会，全链路 no-op） |
| 15 | 判分器自己怎么被验证？ | §3.5 / `run_eval.py --self-test` |
| 16 | 为什么指代消解不在 Router 单测里测？ | §2.2 |
| 17 | 报告概览矩阵的列结构为什么这么设计？ | §3.7（哪列先 ✗ 问题就在哪层） |
| 18 | Router 截断 bug 的根因是什么？谁发现的？ | §4.4（未传 reasoning_effort；评测发现的） |

## 10. 推荐阅读顺序

按"先跑起来看到现象，再读代码理解机制"的顺序，预计 70 分钟（对应 `导学-PickWise.md` 第 3 站）。

| 顺序 | 读什么 | 重点看什么 | 时间 |
| --- | --- | --- | --- |
| 1 | `app/evaluation/cases.json`（随机挑 3 条） | 一条用例长什么样，断言怎么声明 | 5min |
| 2 | `app/evaluation/dataset.py` | 五种字段分组的意图 | 5min |
| 3 | `app/evaluation/trace.py` | `RunTrace` 字段 + `abnormal_llm_calls` 的判据 | 5min |
| 4 | `app/evaluation/sandbox.py` | **重点**：`_instrument` 的 4 个补丁点 + `_guess_purpose` 的演进注释 | 15min |
| 5 | `app/evaluation/metrics.py` | `_CHECKS` 九条 + 五个"防冤"细节 | 10min |
| 6 | `app/evaluation/judges.py` | 防抖动三件套 + v2 修的两个 bug + rubric 里的架构说明 | 10min |
| 7 | `app/evaluation/reporter.py` | **重点**：四个接口 + 三个坑 + no-op 降级 | 12min |
| 8 | `app/scripts/run_eval.py` | `_self_test` + `_print_overview` + 落盘清洁 | 8min |
| 9 | `app/config/settings.py:102-117` | 凭证桥接段（为什么需要） | 3min |
| 10 | `app/multi_agent/orchestrator.py:16,41,57,182-187` | drop-in import + 共享 client + `close()` 的副作用 | 5min |

**读完后建议的动作**（按你的习惯，命令自己跑）：

1. `python app/scripts/run_eval.py --self-test` —— 先确认判分器可信；
2. `python app/scripts/run_eval.py --case-id detail_price` —— 单条跑通，对着 `runs/detail_price.json` 逐字段核对 `RunTrace` 采集是否如你理解；
3. 打开 Langfuse 看板，对照 §5.6 检查树形结构与 score 是否符合预期；
4. 画一遍 §6 的时序草图，标出你仍不确定的步骤，回来继续追问。

---

**关联文档**

- 全局学习路线：`项目学习文档/导学-PickWise.md`
- 模块设计（决策与取舍）：`develop_docs/模块设计/11-评估.md`
- 问题演进记录（今天踩的坑）：`develop_docs/学习笔记/agent评测过程.md`
- RAG 侧的平行记录：`develop_docs/学习笔记/RAG评测过程.md`



# 评测问题
