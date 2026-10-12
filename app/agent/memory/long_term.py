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
    """一条长期记忆事实。"""
    content: str
    category: str  # identity / preference / behavior / issue / other
    created_at: str
    source_session: str = ""


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
            return
        with self.memory_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or data.get("user_id") != self.user_id:
            raise SessionOwnershipError("长期记忆文件归属不匹配")
        if data.get("version") != 1 or not isinstance(data.get("facts"), list):
            raise ValueError("长期记忆文件格式不合法")
        facts = []
        for item in data["facts"]:
            if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                raise ValueError("长期记忆事实格式不合法")
            facts.append(MemoryFact(
                content=item["content"],
                category=item.get("category", "other"),
                created_at=item.get("created_at", ""),
                source_session=item.get("source_session", ""),
            ))
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
            atomic_write_json(self.memory_path, {
                "version": 1,
                "user_id": self.user_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "facts": [asdict(f) for f in self.facts],
                "interaction_summaries": self.interaction_summaries,
            })
            self._pending_facts.clear()
            self._pending_summaries.clear()

    def add_facts(self, new_facts: list[MemoryFact]) -> None:
        """添加新事实，自动去重并裁剪到 max_facts。"""
        with self._lock:
            self._pending_facts.extend(new_facts)
            self._merge_facts(new_facts)

    def _merge_facts(self, new_facts: list[MemoryFact]) -> None:
        existing_contents = {f.content.lower() for f in self.facts}
        for fact in new_facts:
            if fact.content.lower() not in existing_contents:
                self.facts.append(fact)
                existing_contents.add(fact.content.lower())

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

    def extract_and_save(
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
        new_facts, interaction_summary = extract_long_term_facts(
            client, model, messages, summary, list(self.facts),
        )

        if new_facts:
            self.add_facts(new_facts)
        if interaction_summary:
            self.add_interaction_summary(interaction_summary)

        self.save()

    def build_prompt_section(self) -> str | None:
        """生成注入 system prompt 的长期记忆片段。"""
        self.load()
        if not self.facts and not self.interaction_summaries:
            return None

        parts = []
        if self.facts:
            facts_text = "\n".join(
                f"- [{f.category}] {f.content}" for f in self.facts
            )
            parts.append(f"该用户的历史记忆（来自过往会话）：\n{facts_text}")

        if self.interaction_summaries:
            recent = self.interaction_summaries[-3:]
            summaries_text = "\n".join(f"- {s['summary']}" for s in recent)
            parts.append(f"最近的交互记录：\n{summaries_text}")

        return "\n\n".join(parts)

    def reset(self) -> None:
        """清空该用户的长期记忆（文件也删除）。"""
        with self._lock:
            self._load_from_disk()  # 核对归属，不删除其他用户的数据
            self.facts = []
            self.interaction_summaries = []
            self._pending_facts.clear()
            self._pending_summaries.clear()
            self.memory_path.unlink(missing_ok=True)
