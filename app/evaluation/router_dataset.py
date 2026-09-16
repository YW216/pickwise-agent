"""Router 评测数据集：用例结构与加载。

一条 RouterCase 描述「前置对话（可为空）+ 当前用户消息 + 期望场景列表」，
外加评测维度元数据（group / difficulty / description），用于分维度统计。

期望判定：route() 返回的 list[str] 与 expected 完全一致（有序）。
留空 expected 的用例不计分（预留，当前不用）。

与 app/evaluation/dataset.py 风格一致：dataclass + JSON 加载。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RouterCase:
    """单条 Router 评测用例。"""

    id: str
    group: str          # 评测维度：single_presale / single_consult / multi / continue / switch / boundary（2026-09-05 双 Agent 定稿）
    difficulty: str     # easy / medium / hard
    description: str    # 测试点说明（为什么这条用例有价值）
    input: str          # 当前用户消息（待分类）
    expected: list[str]  # 期望场景列表（有序、完全匹配）
    history: list[dict] = field(default_factory=list)  # 前置对话（role/content），可为空
    summary: str | None = None  # 历史压缩摘要（模拟压缩后场景，验证跨压缩指代兜底）


def load_router_cases(path: str | Path) -> list[RouterCase]:
    """从 JSON 加载用例列表，格式 {"cases": [ {RouterCase 字段}, ... ]}。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RouterCase(**item) for item in data["cases"]]
