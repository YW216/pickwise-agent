"""Multi-Agent 编排器：协调 Router、子 Agent、黑板与 Result Agent 完成用户请求。

流程：ContextPack → Router 输出场景列表 → N 个 Agent 并行执行（N=1 特例同路径）
→（N>1）黑板收集 → Result 整合 → full 档合并 → 持久化。

`chat()` 返回纯文本 reply，无响应 schema、无结构化提取调用（设计文档十一节）。
并行阶段只读、合并阶段统一写（设计文档 10.8）：执行期间 raw_messages 一个字不变，
合并由主线程按 SCENARIO_ORDER 固定顺序单点完成。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Optional

#from openai import OpenAI
from langfuse.openai import openai

from app.agent.storage import delete_session, load_session, save_session
from app.agent.compaction import compact, should_compact
from app.agent.context_budget import (
    ContextOverflowError,
    estimate_messages_tokens,
    estimate_text_tokens,
    estimate_tool_definitions_tokens,
    is_transient,
)
from app.agent.product_tracker import format_products_block, merge_products
from app.config.settings import settings
from app.multi_agent.agents import AGENT_CONFIGS, SubAgent
from app.multi_agent.blackboard import BlackboardEntry, render_blackboard
from app.multi_agent.router import DEFAULT_SCENARIO, Router
from app.agent.tools.manager import ToolManager
from app.multi_agent.context_pack import ContextPack, build_working_messages
from app.prompts.agents import RESULT_PROMPT


class MultiAgentOrchestrator:
    """多 Agent 编排器：协调 Router、子 Agent、黑板与 Result Agent。"""

    def __init__(self, session_path: Optional[str] = None):
        self.client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout,
            max_retries=settings.openai_max_retries,
        )
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.session_path = session_path or settings.session_path
        self.context_window = settings.context_window
        self.reserve_tokens = settings.reserve_tokens
        self.compaction_enabled = settings.compaction_enabled
        self.keep_recent_tokens = settings.keep_recent_tokens
        self.summary_max_tokens = settings.summary_max_tokens
        self.summary_max_chars = settings.summary_max_chars
        self.tool_result_max_chars = settings.tool_result_max_chars
        self.max_react_steps = settings.max_react_steps
        self.max_user_input_tokens = settings.max_user_input_tokens

        self.router = Router(self.client, self.model)

        self.agents: dict[str, SubAgent] = {}
        #工具白名单隔离注册
        for key, cfg in AGENT_CONFIGS.items():
            tm = ToolManager(
                use_mcp=settings.mcp_enabled,
                mcp_server_url=settings.mcp_server_url,
                allowed_tools=cfg["tools"],
            )
            self.agents[key] = SubAgent(
                name=cfg["name"],
                tool_manager=tm,
                client=self.client,
                model=self.model,
                temperature=self.temperature,
                tool_result_max_chars=self.tool_result_max_chars,
                context_window=self.context_window,
                reserve_tokens=self.reserve_tokens,
                reasoning_effort=settings.reasoning_effort,
            )

        from app.agent.memory import MemoryManager
        self.memory_manager = MemoryManager(
            client=self.client,
            model=self.model,
            user_id=settings.memory_user_id,
            memory_dir=settings.memory_dir,
            memory_enabled=settings.memory_enabled,
            max_ltm_facts=settings.max_ltm_facts,
        )

        if settings.memory_enabled:
            from app.agent.tools.memory_tool import set_memory_manager
            set_memory_manager(self.memory_manager)

        from app.agent.skills import SkillManager
        self.skill_manager = SkillManager(
            skills_dir=settings.skills_dir,
            enabled=settings.skills_enabled,
        )
        if settings.skills_enabled:
            from app.agent.tools.skill_tool import set_skill_manager
            set_skill_manager(self.skill_manager)

        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None
        # 商品记忆：跨压缩累积的商品 ID/名称/提及价（product_tracker 独立存储）
        self.products: dict = {}
        # 每轮重置；仅用于诊断/评测，不把内部异常写进用户回复。
        self.last_failures: list[dict] = []

        loaded = load_session(self.session_path)
        if loaded:
            self.summary = loaded["summary"]
            self.raw_messages = loaded["messages"]
            self.products = loaded.get("products", {})
            if loaded.get("short_term_memory"):
                self.memory_manager.restore_stm(loaded["short_term_memory"])

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    def chat(self, user_input: str) -> str:
        """主答复与附属工作分离：失败保留用户消息，成功答案不被记忆/落盘覆盖。

        入口拒绝不写历史；已入账的失败轮只补一次最终回复。
        内部错误只记录到 last_failures，不输出日志；用户只看到能力级提示。
        """
        self.last_failures = []
        turn_started = False
        completed = False
        try:
            input_tokens = estimate_text_tokens(user_input)
            if input_tokens > self.max_user_input_tokens:
                return (
                    f"输入过长（约 {input_tokens} token，上限 "
                    f"{self.max_user_input_tokens}），请精简或分段提问。"
                )
            self.raw_messages.append({"role": "user", "content": user_input})
            turn_started = True
            reply = self._chat_inner(user_input)
            completed = True
        except Exception as exc:  # 最外层保护用户体验；错误不靠静默吞掉处理
            reply = self._on_failure(user_input, exc)
            if turn_started:
                self.raw_messages.append({"role": "assistant", "content": reply})

        if turn_started:
            if completed:
                try:
                    self.memory_manager.update_short_term(self.raw_messages[-6:])
                except Exception as exc:
                    self._record_failure("memory", exc, affects_answer=False)
            self._save_safely()
        return reply

    def _record_failure(
        self, stage: str, exc: Exception, *, affects_answer: bool = True,
    ) -> None:
        """记录故障事实，不引入异常继承体系，也不改变现有 client 插桩入口。"""
        if not hasattr(self, "last_failures"):
            self.last_failures = []
        self.last_failures.append({
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "affects_answer": affects_answer,
        })

    def _save_safely(self) -> None:
        """本轮只尝试一次落盘；失败保留内存会话与答复，并明确留下诊断记录。"""
        try:
            self.save()
        except Exception as exc:
            self._record_failure("persistence", exc, affects_answer=False)

    def _on_failure(self, user_input: str, exc: Exception) -> str:
        """只生成失败话术与诊断；历史提交/保存由 chat 单点负责。"""
        self._record_failure("chat", exc)
        agent_key = self._failed_agent_key(exc)
        if agent_key:
            return (
                f"{AGENT_CONFIGS[agent_key]['name']}暂时无法访问，"
                "请稍后重试。"
            )
        return "服务暂时不可用，请稍后重试。"

    @staticmethod
    def _failed_agent_key(exc: Exception) -> Optional[str]:
        """从异常消息里取失败环节的 agent key；取不到返回 None。

        不新增异常子类：单 Agent 失败时上抛的是
        `RuntimeError(f"[{entry.agent}] 执行失败: ...")`（见 _chat_inner），
        已带 `[<key>]` 前缀，直接解析即可——为一次失败引入继承体系不划算。
        """
        text = str(exc)
        for key in AGENT_CONFIGS:
            if text.startswith(f"[{key}] 执行失败:"):
                return key
        return None

    def _chat_inner(self, user_input: str) -> str:
        """只完成路由、专家执行和最终回复提交；记忆/持久化在 chat 外层处理。"""
        pack = self._build_pack()
        # 压缩只针对专家 Agent 的上下文预算（_estimate_context_tokens 估的是
        # build_working_messages + 工具 schema），供后面的专家执行准备。
        if self._try_compact(pack):
            pack = self._build_pack()
        # 路由不再做"强制压缩重试"（2026-10-08 移除，理由见下）。
        #
        # 为什么不重试：Router 根本不读完整历史——它只收「最近 4 条
        # user/assistant（每条 ≤300 字）+ 历史摘要 + 当前输入」，工具结果
        # 完全不进 Router。所以专家上下文变小 ≠ Router 输入变小：
        # 强制压缩多半只是删掉 Router 原本就看不到的消息、同时把摘要换长，
        # 未必缓解 Router 超限，纯属无效功。
        #
        # 保留前面的主动压缩：那是给专家 Agent 用的，与路由无关。
        # 若 Router 真出现超限，正确的做法是限制**它自己**的输入预算
        # （收紧最近条数/摘要长度），而不是拿专家上下文间接补救。
        #
        # SDK 已负责 HTTP 层重试，这里不再叠加传输重试，只做瞬时故障降级。
        try:
            scenarios = self.router.route(user_input, pack.history, summary=pack.summary)
        except Exception as exc:
            if is_transient(exc):
                # 默认咨询可能不能完成推荐诉求：明确记为降级，而不是无损成功。
                self._record_failure("router", exc)
                scenarios = [DEFAULT_SCENARIO]
            else:
                raise
        mode = "single" if len(scenarios) == 1 else "multi"
        names = "、".join(AGENT_CONFIGS[k]["name"] for k in scenarios)
        print(f"\n🔀 [路由] → {names}（场景: {', '.join(scenarios)}，模式: {mode}）")

        entries = self._execute_agents(scenarios, pack, mode)
        for entry in entries:
            if entry.status == "failed":
                self.last_failures.append({
                    "stage": f"agent:{entry.agent}",
                    "error_type": entry.error_type or "AgentExecutionError",
                    "message": entry.error or "专家执行失败",
                    "affects_answer": True,
                })

        if len(entries) == 1:
            entry = entries[0]
            if entry.status == "failed":
                raise RuntimeError(f"[{entry.agent}] 执行失败: {entry.error}")
            # 单 Agent：最终答复直返（new_messages 尾部，10.5 位置约定）
            reply = entry.new_messages[-1]["content"]
            self.raw_messages.extend(entry.new_messages)
        else:
            reply = self._run_result_agent(entries, user_input)
            # full 档合并：各 Agent 工具消息对原样存档（[:-1]，成对约束天然满足），
            # Result 整合文本永远最后一条（设计文档 10.4）
            for entry in entries:
                if entry.status == "success":
                    self.raw_messages.extend(entry.new_messages[:-1])
            self.raw_messages.append({"role": "assistant", "content": reply})

        return reply

    def reset(self):
        self.raw_messages = []
        self.summary = None
        self.products = {}
        self.memory_manager.reset_short_term()
        delete_session(self.session_path)

    def save(self) -> None:
        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
            products=self.products,
        )

    def close(self):
        """长期记忆提取失败不阻止资源清理，也不覆盖已经返回的答案。"""
        try:
            self.memory_manager.consolidate_to_long_term(self.raw_messages, self.summary)
        except Exception as exc:
            self._record_failure("long_term_memory", exc, affects_answer=False)
        finally:
            for agent in self.agents.values():
                try:
                    agent.tool_manager.close()
                except Exception as exc:
                    self._record_failure("tool_close", exc, affects_answer=False)
            try:
                self.client.close()
            except Exception as exc:
                self._record_failure("client_close", exc, affects_answer=False)

    def _build_pack(self) -> ContextPack:
        """构建本轮请求的上下文包（Router 与 Agent 共享的唯一事实来源）。

        skill_catalog 在此留空——技能目录按 Agent 归属各异，由 _execute_agents
        在组装各 Agent 的 working messages 前按 agent_key 过滤注入。
        history 即 raw_messages（工作历史：压缩只保留未压缩部分，
        已压缩段以 summary 承载；6-上下文设计.md 约束 C1/C3）。
        summary（六段叙述）与 product_block（商品记忆块）是两个独立字段，
        由 build_working_messages 分别注入——叙述与结构化数据在 pack 层即分离。
        """
        return ContextPack(
            history=self.raw_messages,
            summary=self.summary,
            memory_sections=self.memory_manager.build_memory_prompt_sections(),
            skill_catalog="",
            product_block=format_products_block(self.products),
        )

    def _skill_catalog_for(self, agent_key: str) -> str:
        """按 Agent 归属过滤技能目录（skill 归属声明在 SKILL.md frontmatter）。"""
        if self.skill_manager and self.skill_manager.enabled:
            return self.skill_manager.build_catalog_prompt(agent_key)
        return ""

    def _execute_agents(
        self, scenarios: list[str], pack: ContextPack, mode: str,
        retry: bool = True,
    ) -> list[BlackboardEntry]:
        """线程池并行执行场景 Agent，返回黑板条目（按固定顺序）。

        完成顺序不定（as_completed），但返回条目按 scenarios 顺序排列
        （Router 已保证 SCENARIO_ORDER），合并可复现（设计文档 13.2）。
        单 Agent 失败抛给调用方；多 Agent 失败记 status=failed，
        由 Result 如实告知，不整体失败（#9 部分失败）。

        Returns:
            BlackboardEntry 列表，长度 = len(scenarios)，顺序与入参一致。
        """

        def run(key: str) -> BlackboardEntry:
            agent = self.agents[key]
            # 技能目录按 Agent 归属过滤（pack 共享只读，per-agent 目录用轻量副本注入）
            agent_pack = replace(pack, skill_catalog=self._skill_catalog_for(key))
            try:
                messages = build_working_messages(agent_pack, AGENT_CONFIGS[key], mode)

                _, new_messages = agent.handle(messages, max_steps=self.max_react_steps)
                if (
                    not new_messages
                    or new_messages[-1].get("role") != "assistant"
                    or new_messages[-1].get("tool_calls")
                    or not isinstance(new_messages[-1].get("content"), str)
                    or not new_messages[-1]["content"].strip()
                ):
                    raise ValueError("专家未返回有效的最终答复")
                return BlackboardEntry(
                    agent=key, status="success", error=None,
                    new_messages=new_messages,
                )
            except ContextOverflowError as exc:
                return BlackboardEntry(
                    agent=key, status="overflow", error=str(exc), new_messages=[],
                    error_type=type(exc).__name__,
                )
            except Exception as e:  # 宽捕获：失败进黑板，不中断其余 Agent
                return BlackboardEntry(
                    agent=key, status="failed", error=str(e), new_messages=[],
                    error_type=type(e).__name__,
                )
                
        #多线程并行执行场景 Agent，按顺序返回结果。   这跟线程池有关
        with ThreadPoolExecutor(max_workers=len(scenarios)) as pool:
            futures = {pool.submit(run, key): key for key in scenarios}
            results: dict[str, BlackboardEntry] = {}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

        # 工作线程全部结束后才做压缩重试；成功结果保留，失败子集最多重试一次。
        overflow = [key for key in scenarios if results[key].status == "overflow"]

        #overflow为有失败的agent，retry表示可以重试（如果是第二次重试，retry是false不再接受重试了）
        #这里就是先验证是否需要和可以处理
        if overflow and retry:
            # 先压，压成了才重试——压不动就没必要重跑（上下文一样大，必然二次溢出）。
            # 注意 _try_compact 不是谓词：成功时它已经删过历史前缀、换过摘要了。
            if self._try_compact(pack, force=True):
                for entry in self._execute_agents(
                    overflow, self._build_pack(), mode, retry=False,
                ):
                    results[entry.agent] = entry
        for entry in results.values():
            if entry.status == "overflow":
                entry.status = "failed"
        return [results[key] for key in scenarios]

    def _run_result_agent(
        self, entries: list[BlackboardEntry], user_input: str,
    ) -> str:
        """Result 读黑板整合；故障/空正文保留专家结论，不把内部异常传给用户。"""
        if not any(entry.status == "success" for entry in entries):
            return self._result_fallback(entries)
        try:
            board_text = render_blackboard(entries, self.tool_result_max_chars)
            messages = [
                {"role": "system", "content": RESULT_PROMPT},
                {"role": "user", "content": (
                    f"用户原始问题：\n{user_input}\n\n各专家工作记录：\n{board_text}"
                )},
            ]
            if estimate_messages_tokens(messages) > self.context_window - self.reserve_tokens:
                raise ContextOverflowError("汇总上下文超过预算")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.reserve_tokens,
                **({"reasoning_effort": settings.reasoning_effort}
                   if settings.reasoning_effort else {}),
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            if not content.strip() or getattr(choice, "finish_reason", None) == "length":
                raise ValueError("汇总未返回完整有效正文")
            return content
        except Exception as exc:
            self._record_failure("result", exc)
            return self._result_fallback(entries)

    @staticmethod
    def _result_fallback(entries: list[BlackboardEntry]) -> str:
        """仅展示已完成结论和能力级失败提示；不输出 traceback/原始异常。"""
        parts = []
        for entry in entries:
            name = AGENT_CONFIGS[entry.agent]["name"]
            content = (
                entry.new_messages[-1].get("content", "")
                if entry.status == "success" and entry.new_messages else ""
            )
            if isinstance(content, str) and content.strip():
                parts.append(f"【{name}】\n{content}")
            else:
                parts.append(f"{name}部分暂时无法回答，请稍后重试。")
        return "\n\n".join(parts) or "服务暂时不可用，请稍后重试。"

    def _estimate_context_tokens(self, pack: ContextPack) -> int:
        """按所有 Agent 中的最大请求估算上下文，覆盖 prompt 与 tool schema。"""
        estimates = []
        for key, cfg in AGENT_CONFIGS.items():
            agent_pack = replace(pack, skill_catalog=self._skill_catalog_for(key))
            for mode in ("single", "multi"):
                messages = build_working_messages(agent_pack, cfg, mode)
                tool_defs = self.agents[key].tool_manager.tool_definitions
                estimates.append(
                    estimate_messages_tokens(messages)
                    + estimate_tool_definitions_tokens(tool_defs)
                )
        return max(estimates, default=0)

    def _try_compact(self, pack: ContextPack, force: bool = False) -> bool:
        """在请求前压缩已完成历史；当前 user turn 永远留在工作历史中。

        事务顺序：先压缩（纯计算）→ 确认能缩小上下文 → 最后才删除前缀并替换
        summary。任何一步失败都保持原状态，零副作用（宁可不压，压缩可重试）。
        pack 复用调用方已构建的上下文包做预算基准（未压缩时调用方直接沿用，
        压缩成功后由调用方重建），避免同一次请求重复构建与估算。
        """
        if not self.compaction_enabled or not self.raw_messages:
            return False

        context_tokens = self._estimate_context_tokens(pack)
        
        # 非强制压缩且上下文未超过预算，直接返回。
        if not force and not should_compact(
            context_tokens, self.context_window, self.reserve_tokens,
        ):
            return False

        active_user_index = len(self.raw_messages) - 1
        if self.raw_messages[active_user_index].get("role") != "user":
            active_user_index = len(self.raw_messages)

        try:
            cut, new_summary = compact(
                messages=self.raw_messages,
                summary=self.summary,
                client=self.client,
                model=self.model,
                keep_recent_tokens=max(1, self.keep_recent_tokens // 2) if force else self.keep_recent_tokens,
                end=active_user_index,
                summary_max_tokens=self.summary_max_tokens,
                summary_max_chars=self.summary_max_chars,
                context_window=self.context_window,
                tool_result_max_chars=self.tool_result_max_chars,
            )
        except Exception as exc:
            self._record_failure("compaction", exc, affects_answer=False)
            return False

        if cut <= 0:
            print("⚠️  [压缩] 没有找到可压缩的完整 turn，跳过本次压缩")
            return False

        # 商品记忆与删除同事务：delta 即将从窗口消失，先并入 store 再删原文。
        delta = self.raw_messages[:cut]
        new_products = merge_products(self.products, delta)
        new_pack = replace(
            pack,
            history=self.raw_messages[cut:],
            summary=new_summary,
            product_block=format_products_block(new_products),
        )
        if self._estimate_context_tokens(new_pack) >= context_tokens:
            print("[压缩] 摘要没有缩小上下文，保留旧状态")
            return False

        # 事务提交点：摘要校验通过且确认缩小上下文后，才真正删除前缀。
        self.products = new_products
        del self.raw_messages[:cut]
        self.summary = new_summary
        print(
            f"\n💾 [已压缩 {cut} 条旧消息 → summary "
            f"({len(new_summary or '')} 字，商品记忆 {len(self.products)} 款，"
            f"保留 {len(self.raw_messages)} 条)]\n"
        )
        return True
