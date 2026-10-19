"""长期记忆：跨会话持久化用户知识。

将用户在多次会话中表现出的偏好、身份、行为模式等提取并持久化为 JSON，
在新会话启动时加载并注入 prompt，让 Agent 具备"记住老客户"的能力。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from threading import Lock, RLock
from weakref import WeakValueDictionary

from openai import OpenAI

from app.agent.memory.extraction import extract_long_term_facts
from app.agent.storage import atomic_write_json
from app.identity import SessionOwnershipError, validate_user_id


_locks = WeakValueDictionary()
_locks_guard = Lock()


def _lock_for(path: Path):
    """不同实例按同一个文件共享锁，闲置锁可随实例回收。"""
    key = os.path.normcase(str(path.resolve()))
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = RLock()
            _locks[key] = lock
        return lock


@dataclass
class MemoryFact:
    """一条用户画像（跨会话）。

    只收四类固定槽位——判据是「下个月还成立、且能改变推荐结果」：
    - constraint 客观约束（身体/环境/兼容性）
    - reject     硬排斥（品牌/形态/型号 + 原因）
    - scene      稳定场景（长期用途，不是本次需求）
    - weight     权重排序（品类内取舍先后）
    预算、候选、"尚未决定"这类窗口参数一律不收（2026-10-19 定稿）。
    """
    slot: str = ""               # constraint / reject / scene / weight
    key: str = ""                # 约束标签、brand:X、品类名
    value: str | list[str] = ""  # 画像值
    quote: str = ""              # 用户原话，便于核对，可为空
    evidence: int = 1            # 被确认次数（同 key 再出现 +1）
    updated_at: str = ""         # 最后一次确认（YYYY-MM-DD）


PROFILE_SLOTS = ("constraint", "reject", "scene", "weight")
_SLOT_TITLES = {"constraint": "约束", "reject": "排斥", "scene": "场景", "weight": "权重"}


def _render_fact(slot: str, fact: MemoryFact) -> str:
    """把一条画像渲染成注入用的短句。

    key 里可能带分组前缀（`brand:星海`），注入时只留可读部分；
    列表值按语义选分隔符：权重用 `>` 表示排序，其余用顿号。
    """
    head = fact.key.split(":", 1)[1] if ":" in fact.key else fact.key
    if isinstance(fact.value, list):
        value = (" > " if slot == "weight" else "、").join(str(item) for item in fact.value)
    else:
        value = str(fact.value or "")
    if not value:
        return head
    if slot == "weight":
        return f"{head} {value}"
    return f"{head}（{value}）"



class LongTermMemory:
    """跨会话长期记忆：持久化用户知识。"""

    def __init__(
        self,
        user_id: str = "default",
        memory_dir: str = "app/sessions/memory",
        max_facts: int = 50,
    ):
        self.user_id = validate_user_id(user_id)
        self.memory_dir = Path(memory_dir).resolve()
        self.max_facts = max_facts
        self.facts: list[MemoryFact] = []
        self.interaction_summaries: list[dict] = []
        # 旧版（version 1）的 facts 原样保留在同一文件里：不迁移、不注入，
        # 也不删除——用户已明确不做数据清洗（2026-10-19）。
        self._legacy_facts: list[dict] = []
        self._pending_facts: list[MemoryFact] = []
        self._pending_summaries: list[dict] = []
        self._lock = _lock_for(self.memory_path)

    @property
    def memory_path(self) -> Path:
        path = (self.memory_dir / f"{self.user_id}.json").resolve()
        if not path.is_relative_to(self.memory_dir):
            raise ValueError("记忆路径超出存储目录")
        return path

    def load(self) -> None:
        """刷新最新快照，同时保留本实例尚未提交的增量。"""
        with self._lock:
            self._load_from_disk()
            self._merge_facts(self._pending_facts)
            self._merge_summaries(self._pending_summaries)

    def _load_from_disk(self) -> None:
        if not self.memory_path.exists():
            self.facts = []
            self.interaction_summaries = []
            self._legacy_facts = []
            return
        with self.memory_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or data.get("user_id") != self.user_id:
            raise SessionOwnershipError("长期记忆文件归属不匹配")
        if data.get("version") not in (1, 2):
            raise ValueError("长期记忆文件格式不合法")

        profile = data.get("profile")
        if profile is not None and not isinstance(profile, list):
            raise ValueError("长期记忆画像格式不合法")
        facts = []
        for item in profile or []:
            if not isinstance(item, dict) or not isinstance(item.get("key"), str):
                raise ValueError("长期记忆画像条目格式不合法")
            value = item.get("value", "")
            if not isinstance(value, (str, list)):
                raise ValueError("长期记忆画像取值格式不合法")
            facts.append(MemoryFact(
                slot=item.get("slot", ""),
                key=item["key"],
                value=value,
                quote=item.get("quote", ""),
                evidence=int(item.get("evidence", 1) or 1),
                updated_at=item.get("updated_at", ""),
            ))
        self.facts = facts

        # version 1 的 facts 只保留原样，不再参与注入。
        legacy = data.get("facts")
        self._legacy_facts = legacy if isinstance(legacy, list) else []
        summaries = data.get("interaction_summaries", [])
        if not isinstance(summaries, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("summary"), str)
            for item in summaries
        ):
            raise ValueError("长期记忆摘要格式不合法")
        self.facts = facts
        self.interaction_summaries = summaries

    def save(self) -> None:
        """持锁读最新文件、合并增量再写入，避免跨窗口覆盖。"""
        with self._lock:
            self.load()
            payload = {
                "version": 2,
                "user_id": self.user_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "profile": [asdict(f) for f in self.facts],
                "interaction_summaries": self.interaction_summaries,
            }
            if self._legacy_facts:
                payload["facts"] = self._legacy_facts  # 旧数据原样带回去，不丢也不读
            atomic_write_json(self.memory_path, payload)
            self._pending_facts.clear()
            self._pending_summaries.clear()

    def add_facts(self, new_facts: list[MemoryFact]) -> None:
        """写入画像：同 (slot, key) 覆盖旧值，其余追加到末尾。"""
        with self._lock:
            self._pending_facts.extend(new_facts)
            self._merge_facts(new_facts)

    def _merge_facts(self, new_facts: list[MemoryFact]) -> None:
        """按 (slot, key) upsert。

        冲突语义（2026-10-19 定稿）：同一个 key 只保留一条有效值——先删旧的，
        新的追加到末尾；`evidence` 累加表示"被确认过几次"，值本身以最新为准。
        这样文件里不会出现"预算 8000"和"预算 10000"并存。
        """
        today = datetime.now().date().isoformat()
        for fact in new_facts:
            if not fact.slot or not fact.key:
                continue  # 结构不全的条目直接丢，宁可漏不可错
            fact.updated_at = fact.updated_at or today
            previous = next(
                (item for item in self.facts if (item.slot, item.key) == (fact.slot, fact.key)),
                None,
            )
            if previous is not None:
                self.facts.remove(previous)
                fact.evidence = max(1, int(previous.evidence or 1)) + 1
            else:
                fact.evidence = max(1, int(fact.evidence or 1))
            self.facts.append(fact)

        # 超出上限丢最旧的：文件里新的永远在后面，所以从头部截。
        if len(self.facts) > self.max_facts:
            self.facts = self.facts[-self.max_facts:]

    def add_interaction_summary(self, summary: str) -> None:
        with self._lock:
            item = {"summary": summary, "timestamp": datetime.now().isoformat(timespec="seconds")}
            self._pending_summaries.append(item)
            self._merge_summaries([item])

    def _merge_summaries(self, summaries: list[dict]) -> None:
        for item in summaries:
            if item not in self.interaction_summaries:
                self.interaction_summaries.append(item)

    async def extract_and_save(
        self,
        client: OpenAI,
        model: str,
        messages: list[dict],
        summary: Optional[str],
    ) -> None:
        """从会话消息中提取长期记忆事实并保存。"""
        if not messages and not summary:
            return

        self.load()
        new_facts, interaction_summary = await extract_long_term_facts(
            client, model, messages, summary, list(self.facts),
        )

        if new_facts:
            self.add_facts(new_facts)
        if interaction_summary:
            self.add_interaction_summary(interaction_summary)

        self.save()

    def build_prompt_section(self) -> str | None:
        """生成注入 system prompt 的用户画像片段。

        只注入四类画像，**不注入 interaction_summaries**——那是对会话日志，
        混进来会让模型把"上次聊到哪"当成"用户说过什么"（2026-10-19 定稿）。
        """
        self.load()
        if not self.facts:
            return None

        parts = [
            # 这段头必须写死"不是本次对话的内容"。旧的「该用户的历史记忆」措辞
            # 会被模型当成会话内已知信息，新开会话第一轮就说"按你之前说过的
            # 预算和玩法"——用户从没在新会话里说过（实测 2026-10-19）。
            "以下是该用户**过往会话**留下的画像，可能已过时，"
            "且**不属于本次对话**（用户在本会话中并未说过这些）。使用约束：\n"
            "① 不得表述成「你之前说过」「你刚才提到」；需要引用时说明是历史偏好；\n"
            "② 与本轮用户明确要求冲突时，一律以本轮为准；\n"
            "③ 报价必须用工具查到的现价，历史价仅可作叙述。",
        ]
        lines = []
        for slot in PROFILE_SLOTS:
            items = [f for f in self.facts if f.slot == slot]
            if not items:
                continue
            rendered = "；".join(_render_fact(slot, fact) for fact in items)
            lines.append(f"{_SLOT_TITLES[slot]}：{rendered}")
        if lines:
            parts.append("【用户画像】\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def reset(self) -> None:
        """清空该用户的长期记忆（文件也删除）。"""
        with self._lock:
            self._load_from_disk()  # 核对归属，不删除其他用户的数据
            self.facts = []
            self.interaction_summaries = []
            self._legacy_facts = []
            self._pending_facts.clear()
            self._pending_summaries.clear()
            self.memory_path.unlink(missing_ok=True)
