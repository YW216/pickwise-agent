# PickWise（ecom-service-agent-main）项目导学

> 项目定位：教学型电商选购助手「小P」，双 Agent（售前/咨询）协作架构 + 双层自动化评估体系。
> 本导学基于 2026-09-16 源码实测梳理，所有路径均可在仓库中定位。

## 1. 前置知识（面试高频标注）

| 知识点 | 为何需要 | 在本项目中的位置 | 高频度 |
| --- | --- | --- | --- |
| OpenAI Function Calling 协议 | 整个工具系统的通信基础（tool_calls / tool 消息对） | `app/agent/tools/registry.py`、`app/multi_agent/agents.py` handle() | ⭐⭐⭐ |
| ReAct 循环 | 子 Agent 的执行模型（思考→行动→观察循环） | `app/multi_agent/agents.py` SubAgent.handle() | ⭐⭐⭐ |
| 多 Agent 编排（路由→并行→汇总） | 项目核心架构，面试必问 | `app/multi_agent/orchestrator.py` | ⭐⭐⭐ |
| 上下文窗口管理 / 会话压缩 | Agent 工程核心难题，强亮点 | `app/agent/compaction.py`、`context_budget.py` | ⭐⭐⭐ |
| LLM 评估（规则断言 + LLM-as-judge） | 自动化评估体系是近年高频考点 | `app/evaluation/` 全目录 | ⭐⭐⭐ |
| RAG：Embedding + 向量库 + 混合检索 | 知识问答的能力底座 | `app/agent/rag/`（retriever/embedder/milvus_backend） | ⭐⭐ |
| 记忆系统（短期/长期） | 个性化推荐与多轮体验 | `app/agent/memory/` | ⭐⭐ |
| Agent Skills 渐进加载（Anthropic 开放标准） | 上下文经济性的典型案例 | `app/agent/skills/loader.py` | ⭐⭐ |
| 线程池并行与 Copy-on-Write | 并行安全的工程保障 | `orchestrator._execute_agents`、`agents.handle` | ⭐⭐ |
| pydantic-settings 配置管理 | 工程化基础 | `app/config/settings.py` | ⭐ |

## 2. 重点亮点与学习顺序（先看这个）

| # | 亮点标题 | 为什么重要 | 通用技术关键词 | 先看哪些文件 | 建议顺序 |
| --- | --- | --- | --- | --- | --- |
| 1 | 多 Agent 编排与并行安全 | 一条消息多个诉求如何被并行处理又不互相污染上下文 | 意图路由、并行扇出、黑板模式、部分失败容错 | `orchestrator.py` → `router.py` → `agents.py` → `blackboard.py` | 第 1 站 |
| 2 | 上下文工程：预算预检 + 事务式压缩 + 商品记忆 | 解决无限对话撑爆窗口，且压缩不丢结构化关键信息 | token 预算、摘要压缩、事务顺序（先摘要后删除） | `context_budget.py` → `compaction.py` → `product_tracker.py` | 第 2 站 |
| 3 | 双层自动化评估体系 | 没有评估就没有迭代依据；规则断言保确定性、LLM judge 做过程评分 | 评测沙箱、RunTrace、规则断言、LLM-as-judge（advisory） | `evaluation/sandbox.py` → `metrics.py` → `judges.py` → `evaluator.py` | 第 3 站 |
| 4 | 工具白名单隔离与信封协议 | 每 Agent 只见自己的工具；工具结果统一信封防协议违约 | 最小权限、统一响应信封、存档/视图分离 | `tools/manager.py` → `tools/registry.py` → `tools/result.py` | 第 4 站 |
| 5 | RAG 混合检索（dense + BM25 + RRF） | 纯语义检索召回不稳，混合检索是业界主流方案 | hybrid search、RRF 融合、milvus-lite/standalone 同构切换 | `rag/retriever.py` → `rag/milvus_backend.py` → `rag/embedder.py` | 第 5 站 |
| 6 | 记忆与技能系统 | 短期/长期记忆分工 + 技能目录渐进注入，都是上下文经济性设计 | STM/LTM 巩固、frontmatter 元数据、按需加载 | `memory/manager.py` → `skills/loader.py` | 第 6 站 |

## 3. 必备知识点

- [ ] tool_calls 与 tool 消息必须成对（OpenAI 协议硬约束）
- [ ] ReAct = 思考/行动/观察循环；超过最大步数时"去工具兜底"收尾
- [ ] 单 Agent 与多 Agent 模式统一走并行执行器（N=1 是特例，直返不加黑板）
- [ ] raw_messages（存档全量）与 working messages（工作视图截断）是两条轨道
- [ ] 压缩事务：先算摘要 → 确认能缩小 → 才删前缀，任何失败零副作用
- [ ] 商品记忆独立于文本摘要（结构化存 ID/名称/提及价）
- [ ] Router 输出契约：有序、去重、合法、非空、兜底 consult
- [ ] 工具结果统一信封 `{success, error, data}`，经序列化进 tool 消息
- [ ] 黑板：共同写不互读，只有 Result Agent 读全量；单轮即弃
- [ ] LLM judge 分数只做 advisory，不参与 passed 判定

## 4. 推荐阅读（结合仓库）

| 主题               | 通用技术点                       | 建议阅读位置                                                                                      | 预计时间   | 读完能回答什么                                  |
| ---------------- | --------------------------- | ------------------------------------------------------------------------------------------- | ------ | ---------------------------------------- |
| 主链路一次请求的生命周期     | 编排、路由、并行、合并                 | `main.py` + `app/multi_agent/orchestrator.py` chat()                                        | 40min  | 一条消息从进入到返回经过哪几步？N=1 和 N>1 分叉在哪？          |
| 路由器设计            | LLM 分类、输出解析兜底、多轮上下文         | `app/multi_agent/router.py` + `app/prompts/agents.py` ROUTER_PROMPT                         | 30min  | 为什么固定 SCENARIO_ORDER？max_tokens 为何是 512？ |
| 子 Agent ReAct 循环 | Copy-on-Write、溢出上抛、步数兜底     | `app/multi_agent/agents.py`                                                                 | 40min  | 并行为什么安全？溢出为何上抛而非就地压缩？                    |
| 上下文预算与压缩         | token 估算、事务式压缩、六段摘要         | `app/agent/context_budget.py` + `compaction.py` + `summarizer.py`                           | 60min  | 压缩怎么保证不丢当前 turn？摘要失败会怎样？                 |
| 商品记忆             | 结构化记忆独立存续                   | `app/agent/product_tracker.py`                                                              | 20min  | 压缩后商品信息如何存活？历史价如何防误导？                    |
| 工具系统             | 注册表、白名单、信封协议                | `app/agent/tools/registry.py` + `manager.py` + `result.py`                                  | 40min  | 工具执行失败模型看到什么？为什么要信封？                     |
| RAG 混合检索         | embedding、BM25、RRF、索引模型校验   | `app/agent/rag/retriever.py` + `milvus_backend.py`                                          | 50min  | 为什么要混合检索？换了 embedding 模型会发生什么？           |
| 记忆系统             | STM 提取、LTM 巩固、prompt 注入     | `app/agent/memory/manager.py` + `short_term.py` + `long_term.py`                            | 40min  | 记忆何时写入 prompt？会话结束发生什么？                  |
| 技能系统             | 渐进披露、frontmatter、Agent 归属过滤 | `app/agent/skills/loader.py` + `skills/definitions/*/SKILL.md`                              | 30min  | 技能目录怎么省 token？按 Agent 过滤在哪实现？            |
| 评估体系             | 沙箱、RunTrace、规则断言、LLM judge  | `app/evaluation/`（sandbox/trace/metrics/judges/evaluator）+ `app/scripts/run_router_eval.py` | 70min  | passed 怎么判定？judge 分数起什么作用？               |
| 自己的设计文档          | 全局设计决策与演进史                  | `develop_docs/模块设计/`（16 篇，重点 4/6/8/11）                                                      | 120min | 每个模块当时为什么这样取舍？                           |

## 5. 自学提醒

- 若某文件或原理看不懂，请继续追问 AI；本导学只负责给学习路径与题目，不提供逐行讲解。
- 建议每站读完画一遍调用链草图，再跑一次 `python main.py` 对照实际输出验证理解（长评测脚本你自己跑）。

## 6. 项目技术定位

**AI 应用后端 / Agent 工程**：纯 Python 编排 LLM（DeepSeek）+ Function Calling + Milvus 向量检索 + 自动化评估，无传统 Web 前端；CLI 入口只是交互壳。

## 7. 核心原理解析

1. **一条消息多个诉求怎么办**
   问题：用户一句话既要求推荐又问知识，单 Agent 串行处理慢且 prompt 臃肿。
   机制：Router 输出场景列表 → ThreadPool 并行执行对应 Agent → 黑板收集回执 → Result Agent 整合统一回复；单场景时直返不加黑板。
   落点：`orchestrator.chat()` → `_execute_agents()` → `_run_result_agent()`。

2. **上下文无限增长怎么办**
   问题：多轮 + 工具轨迹会让 prompt 超出窗口。
   机制：请求前估算（消息 + 工具 schema 双估），超线触发压缩；压缩是无状态纯函数，事务顺序"先摘要→确认缩小→后删除"，失败零副作用；溢出异常统一上抛由编排器压缩后整体重试一次。
   落点：`context_budget.py` 预检、`compaction.py` 切点与摘要、`orchestrator._try_compact()` 提交。

3. **压缩会丢关键信息怎么办**
   问题：文本摘要擅长叙事但保不住结构化数据（商品 ID、价格）。
   机制：商品记忆独立存储（product_tracker），压缩同事务先把 delta 并入 store 再删原文；摘要价格仅作叙述，报价前必须工具查现价。
   落点：`product_tracker.py` merge_products、`orchestrator._try_compact()` 商品同事务、PRESALE_PROMPT 回复规范第 6 条。

4. **<font color="#ff0000">工具结果撑爆上下文怎么办</font>**
   问题：一次检索返回上万字符，几轮就超窗。
   机制：双轨制——存档轨道（raw_messages/new_messages）保留全量可回放，工作轨道（working）按 tool_result_max_chars 截断视图；两边同源不脱敏事实。
   落点：`agents.handle()` 中 tool_result_view 的调用、`context_budget.py`。

5. **LLM 输出不可靠怎么办**
   问题：路由可能输出非法词/乱序；工具结果塞 dict 违反协议。
   机制：Router 解析层做小写化/去重/固定顺序/兜底默认场景；工具结果统一信封 + 序列化为合法字符串；工具白名单按 Agent 隔离，最小权限。
   落点：`router._parse()`、`tools/result.py` + `execute_tool_as_message`、`manager._filter_tools()`。

6. **怎么知道系统真的变好了**
   问题：改 prompt/换模型后无法验证是否回归。
   机制：三层用例（路由 55 条 / 端到端 20 条 / RAG 152 条）+ 沙箱隔离运行 + 确定性规则断言判 passed + LLM judge 过程评分（advisory）+ RunTrace 留痕 + 报告落盘。
   落点：`evaluation/` 全目录、`app/scripts/run_router_eval.py` 等三个入口、`app/evaluation/runs/` 历史报告。

## 8. 关键设计决策

| 决策 | 备选 | 取舍 | 风险 | 验证 |
| --- | --- | --- | --- | --- |
| 单/多模式统一并行执行器 | 两套代码路径分写 | 多 20 行代码换路径一致性，行为可预测 | 无明显 | test_multi_agent / test_p3_multi |
| 压缩先摘要后删除（事务式） | 直接删再补摘要 | 压缩可能失败但绝不破坏现场 | 极端时宁可不压 | test_compaction |
| 工具结果双轨制（全量存档/截断工作视图） | 单轨截断（丢存档）或单轨全量（爆窗） | 存档可回放、工作视图可控 | 存档仍占窗口，靠压缩兜底 | test_p2_context |
| Result Agent 不调工具 | 允许 Result 再检索 | 角色单一、上下文可控、避免循环 | 汇总缺信息只能如实说 | RESULT_PROMPT 约束 |
| Router max_tokens=512 且不传 reasoning | 小预算 + 深思考 | 实测小预算被思考吃光导致 100% 兜底 | — | 报告记录在 router.py 注释 |
| LLM judge 只做 advisory | judge 分数参与硬判定 | 规则断言可复现，judge 供过程洞察 | judge 误判不影响通过率 | 11-评估.md 第三节 |
| 黑板共同写不互读 | Agent 互读协商 | 简单并行、无死锁、延迟低 | 跨 Agent 冲突靠 Result 调解 | blackboard.py 设计注释 |

## 9. 量化与验证（含待测，建议）

- 路由准确率：55 条用例 100%（`app/evaluation/runs/router_report.md`，2026-09-08），建议每次改 ROUTER_PROMPT 后回归。
- 端到端 20 条 + RAG 152 条：报告在 `report/` 与 `runs/`，建议跑 `run_eval.py` / `run_rag_eval.py` 刷新（长时间跑批自行执行）。
- 口碑检索冗余改造（历史成果）：调用 14→5 次、token -39%，属于早期版本数据，当前 retrieve_reviews 已下线，面试表述需注明版本背景（待补：当前架构下的等价指标）。
- 压缩效果：摘要是否必然缩小上下文有断言保护，但压缩节省率未系统统计（待测）。
- 并行加速比：多场景并行的时延收益未测量（待测）。
