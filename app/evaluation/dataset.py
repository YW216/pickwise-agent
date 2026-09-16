"""评估数据集：端到端用例的数据结构与加载（P6）。

一条 EvalCase 描述「输入是什么」+「期望表现是什么」，期望全部为规则可判的
确定性断言（见 11-评估.md 第三节）：

- 归因断言：expected_route（路由，仅用于失败定位）、expected_tools（工具行为）。
- 质量断言：reply_prices_within（价格 ∈ 目录且 ≤ 上限）、
  must_include_catalog_price / _any（精确价命中）、valid_ids_only（ID 不编造）、
  keyword_hits（政策关键词）、min_chars（最低篇幅）、reply_prices_in_catalog
  （无编造价格）。

留空的期望项在评分时跳过（不计分），避免不适用断言拖垮聚合。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    """单条端到端评估用例。turns 支持多轮，按顺序喂给同一个编排器实例。"""

    id: str
    category: str  # 用例分类：主流程 / 对比 / 咨询 / 跨轮指代 / 边界（报告分组用）
    description: str
    turns: list[str]  # 多轮输入，单轮即 len==1

    # ---------- 归因断言 ----------
    expected_route: list[str] | None = None  # 每轮期望路由（与 turns 对齐）；每轮为场景名，
    # 支持 "presale|consult" 表示二选一皆可；None=不判
    expected_tools: list[str] | None = None  # any-of：至少调用其一；空列表=必须零工具调用；None=不判

    # ---------- 质量断言：数值保真 ----------
    reply_prices_within: dict | None = None  # {"category": "耳机", "max_price": 1500, "last_turn_only": true}
    must_include_catalog_price: dict | None = None  # {"LP-01": 5180}——ID 与目录价都必须出现在回复中
    must_include_catalog_price_any: dict | None = None  # 同上，但命中任意一对即可（多轮指代用）
    valid_ids_only: bool = False  # 回复中的商品 ID 必须存在于 mock 目录
    reply_prices_in_catalog: bool = False  # 回复中的价格必须都存在于目录（无编造价格）

    # ---------- 质量断言：内容 ----------
    keyword_hits: list[str] = field(default_factory=list)  # 最终回复应命中的关键词
    min_chars: int | None = None  # 最终回复最低字符数

    # ---------- 主观质量（LLM judge，advisory：不参与 pass/fail） ----------
    judge_aspects: list[str] = field(default_factory=list)  # "answer_quality" / "process"；空=不判


def load_dataset(path: str | Path) -> list[EvalCase]:
    """从 JSON 文件加载用例列表，文件格式为 {"cases": [ {EvalCase 字段}, ... ]}。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in data["cases"]]
