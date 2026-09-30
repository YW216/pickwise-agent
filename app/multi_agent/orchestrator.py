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
    is_context_overflow,
)
from app.agent.product_tracker import format_products_block, merge_products
from app.config.settings import settings
from app.multi_agent.agents import AGENT_CONFIGS, SubAgent
from app.multi_agent.blackboard import BlackboardEntry, render_blackboard
from app.multi_agent.router import Router
from app.agent.tools.manager import ToolManager
from app.multi_agent.context_pack import ContextPack, build_working_messages
from app.prompts.agents import RESULT_PROMPT


class MultiAgentOrchestrator:
    """多 Agent 编排器：协调 Router、子 Agent、黑板与 Result Agent。"""

    def __init__(self, session_path: Optional[str] = None):
        self.client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
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
        """路由 → 场景 Agent 并行执行 →（多场景）Result 整合 → 返回回复文本。

        N=1 与 N>1 统一走并行执行器（N=1 是特例：线程池跑 1 个任务，
        无黑板、无 Result——Agent 最终文本直返，new_messages 全量合并，A 现状）。
        """
        # 入口体量闸门：超长输入必须挡在进入历史之前。当前 user 消息永远不在
        # 压缩范围内（_try_compact 的 active_user_index 把它排除在外），一旦入账，
        # "输入本身超长"就没有任何机制能救——压缩删的是它之前的历史。
        # 返回提示而非抛异常：这是可恢复的输入问题，不是系统故障；且不写历史、
        # 不调 LLM，用户直接换个说法重试即可。
        # （入口的完整校验——prompt 注入等——另立专项，此处只做体量。）
        input_tokens = estimate_text_tokens(user_input)
        if input_tokens > self.max_user_input_tokens:
            return (
                f"输入过长（约 {input_tokens} token，上限 "
                f"{self.max_user_input_tokens}），请精简或分段提问。"
            )

        self.raw_messages.append({"role": "user", "content": user_input})
        pack = self._build_pack()
        if self._try_compact(pack):
            pack = self._build_pack()  # summary 已更新、旧消息已删除，重建上下文包
        try:
            scenarios = self.router.route(user_input, pack.history, summary=pack.summary)
        except Exception as exc:
            if not is_context_overflow(exc) or not self._try_compact(pack, force=True):
                raise
            pack = self._build_pack()
            scenarios = self.router.route(user_input, pack.history, summary=pack.summary)
        mode = "single" if len(scenarios) == 1 else "multi"
        names = "、".join(AGENT_CONFIGS[k]["name"] for k in scenarios)
        print(f"\n🔀 [路由] → {names}（场景: {', '.join(scenarios)}，模式: {mode}）")

        entries = self._execute_agents(scenarios, pack, mode)

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

        self.memory_manager.update_short_term(self.raw_messages[-6:])

        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
            products=self.products,
        )
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
        self.memory_manager.consolidate_to_long_term(
            self.raw_messages, self.summary,
        )
        for agent in self.agents.values():
            agent.tool_manager.close()

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

                _, new_messages = agent.handle(messages, max_steps=self.max_react_steps)  #实际执行的位置
                return BlackboardEntry(
                    agent=key, status="success", error=None,
                    new_messages=new_messages,
                )
            except ContextOverflowError as exc:
                return BlackboardEntry(
                    agent=key, status="overflow", error=str(exc), new_messages=[],
                )
            except Exception as e:  # 宽捕获：失败进黑板，不中断其余 Agent
                print(f"  ⚠️ [{agent.name}] 执行失败: {e}")
                return BlackboardEntry(
                    agent=key, status="failed", error=str(e), new_messages=[],
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
        """Result Agent：读黑板全部条目，整合成统一回复（调 LLM、不调工具）。

        输入 = 黑板全量轨迹文本渲染（决策 #13）+ 用户原始问题；
        输出 = 一条面向用户的回复文本（8.2 prompt 九条约束）。
        """
        board_text = render_blackboard(entries, self.tool_result_max_chars)
        messages = [
            {"role": "system", "content": RESULT_PROMPT},
            {"role": "user", "content": (
                f"用户原始问题：\n{user_input}\n\n各专家工作记录：\n{board_text}"
            )},
        ]
        try:
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
            return response.choices[0].message.content or ""
        except Exception as exc:
            if not isinstance(exc, ContextOverflowError) and not is_context_overflow(exc):
                raise
            print("[汇总] 上下文超限，返回已有结论")
            return "\n\n".join(
                entry.new_messages[-1]["content"] if entry.status == "success"
                else f"{AGENT_CONFIGS[entry.agent]['name']}部分暂时无法回答。"
                for entry in entries
            )

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
            print(f"⚠️  [压缩] 失败，保留原上下文: {exc}")
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
