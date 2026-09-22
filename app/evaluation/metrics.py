"""规则判分库：对 RunTrace 应用用例的确定性断言（P6）。

每个 check_* 函数对应一种断言，返回 CheckResult（是否适用 + 是否通过 + 明细）。
全部判分只依赖本地 mock 目录（ground truth 确定），不调用 LLM——基线可复现。

判分范围约定：
- valid_ids_only / reply_prices_within / must_include_* 默认检查全部轮次的回复拼接；
  reply_prices_within 支持 last_turn_only（中途改预算场景：上限只约束最后一轮）。
- keyword_hits / min_chars 检查最终回复。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.db.snapshot import PRODUCTS
from app.evaluation.dataset import EvalCase
from app.evaluation.trace import RunTrace


@dataclass
class CheckResult:
    """单条断言的判定结果。applicable=False 表示该用例未指定此断言（不计分）。"""

    name: str
    applicable: bool
    passed: bool | None = None
    detail: str = ""

    @classmethod
    def skip(cls, name: str) -> "CheckResult":
        return cls(name=name, applicable=False)

    @classmethod
    def of(cls, name: str, passed: bool, detail: str = "") -> "CheckResult":
        return cls(name=name, applicable=True, passed=passed, detail=detail)


# ---------- 目录索引（ground truth） ----------

_PRODUCT_BY_ID = dict(PRODUCTS)  # PRODUCTS 本身就是 {product_id: 商品} 映射
_CATEGORY_PRICES: dict[str, set[int]] = {}
for _p in _PRODUCT_BY_ID.values():
    _CATEGORY_PRICES.setdefault(_p["category"], set()).add(_p["price"])

# 边界用否定环视而非 \b：中文与 ID 相邻时 \b 不成立（\w 含中文），会漏检，
# 使数值保真 / 幻觉断言静默放宽。与 product_tracker._PRODUCT_ID_PATTERN 语义一致。
_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:LP|PH|HP)-\d{2}(?!\d)")
# 价格形如 ¥5180 / 5180 元（要求货币标记，避免把"16GB""14寸"这类规格数字误当价格）
_PRICE_PATTERN = re.compile(r"[¥￥]\s*(\d{3,6})|(\d{3,6})\s*元")


def _normalize(text: str) -> str:
    """去掉千分位逗号——LLM 常写 ¥5,180，目录价是 5180。"""
    return text.replace(",", "")


def _extract_prices(text: str) -> list[int]:
    values = []
    for match in _PRICE_PATTERN.finditer(_normalize(text)):
        value = int(match.group(1) or match.group(2))
        values.append(value)
    return values


# ---------- 断言实现 ----------

def check_route(case: EvalCase, trace: RunTrace) -> CheckResult:
    """每轮路由结果与期望一致（归因断言；期望按轮给场景名，兼容给列表）。"""
    if case.expected_route is None:
        return CheckResult.skip("route")
    if len(trace.routes) != len(case.expected_route):
        return CheckResult.of(
            "route", False,
            f"路由轮数 {len(trace.routes)} != 期望 {len(case.expected_route)}",
        )
    for i, (actual, expected) in enumerate(zip(trace.routes, case.expected_route)):
        # "presale|consult" = 本轮二选一皆可（模糊输入的合理路由不止一种）
        acceptable = [
            alt.strip().split("/") if isinstance(alt, str) else list(alt)
            for alt in str(expected).split("|")
        ] if isinstance(expected, str) else None
        if isinstance(expected, str):
            if list(actual) not in acceptable:
                return CheckResult.of(
                    "route", False,
                    f"第 {i + 1} 轮路由 {actual} 不在可接受范围 {acceptable}",
                )
        elif list(actual) != list(expected):
            return CheckResult.of(
                "route", False,
                f"第 {i + 1} 轮路由 {actual} != 期望 {list(expected)}",
            )
    return CheckResult.of("route", True, "，".join("/".join(r) for r in trace.routes))


def successful_tool_names(trace: RunTrace) -> list[str]:
    """至少一次成功执行（结果信封 success=true）的工具名。

    调用失败（success=false，如查询无结果）不算"信息需求被满足"——
    recommend_reviews 的教训：两次 0 结果的错误调用不应命中 expected_tools。
    result 形态兼容两种：沙箱插桩采集的 dict（execute_tool 返回值）
    与序列化后的 JSON 串。
    """
    names = []
    for obs in trace.tool_observations:
        payload = obs.result
        if not isinstance(payload, dict):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        if isinstance(payload, dict) and payload.get("success") is True:
            names.append(obs.name)
    return names


def check_tools(case: EvalCase, trace: RunTrace) -> CheckResult:
    """工具行为：非空列表=any-of 至少命中其一；空列表=必须零工具调用；None=不判。

    命中口径由 case.tools_success_required 决定：
    - True（默认）：至少一次「成功」调用（success=true）——失败调用不算信息需求被满足；
    - False：只要求「发起过调用」——用于查无结果类用例（工具合法返回 success=false，
      按成功口径判会让正确行为必挂，no_match_honesty 的教训）。
    """
    if case.expected_tools is None:
        return CheckResult.skip("tools")
    called = trace.tool_call_names
    succeeded = successful_tool_names(trace)
    if case.expected_tools:
        hits = (set(case.expected_tools) & set(succeeded)) if case.tools_success_required \
            else (set(case.expected_tools) & set(called))
        if not hits:
            criterion = "成功调用" if case.tools_success_required else "调用（失败也算）"
            return CheckResult.of(
                "tools", False,
                f"期望工具均未{criterion}：{case.expected_tools}"
                f"（实际调用：{called or '无'}，其中成功：{sorted(set(succeeded)) or '无'}）",
            )
        note = "" if case.tools_success_required else "（查无结果类用例：只判是否发起调用）"
        return CheckResult.of("tools", True, f"命中 {sorted(hits)}{note}")
    if called:
        return CheckResult.of("tools", False, f"不应调用工具，实际调用了：{called}")
    return CheckResult.of("tools", True, "未调用工具（符合预期）")


def check_valid_ids(case: EvalCase, trace: RunTrace) -> CheckResult:
    """回复中的商品 ID 必须存在于 mock 目录（防幻觉核心断言）。"""
    if not case.valid_ids_only:
        return CheckResult.skip("valid_ids")
    text = _normalize(trace.all_replies_text)
    mentioned = sorted(set(_ID_PATTERN.findall(text)))
    fabricated = [pid for pid in mentioned if pid not in _PRODUCT_BY_ID]
    if fabricated:
        return CheckResult.of("valid_ids", False, f"编造的商品 ID：{fabricated}")
    return CheckResult.of("valid_ids", True, f"提到的 ID 均合法：{mentioned or '（无）'}")


def check_prices_within(case: EvalCase, trace: RunTrace) -> CheckResult:
    """回复中的价格必须属于指定品类目录且 ≤ 上限。"""
    spec = case.reply_prices_within
    if not spec:
        return CheckResult.skip("prices_within")
    category = spec["category"]
    max_price = spec["max_price"]
    last_turn_only = spec.get("last_turn_only", False)
    text = trace.replies[-1] if last_turn_only else trace.all_replies_text

    valid_prices = _CATEGORY_PRICES.get(category, set())
    catalog_prices = [v for v in _extract_prices(text) if v in valid_prices]
    if not catalog_prices:
        return CheckResult.of(
            "prices_within", True,
            f"未提及 {category} 目录价（预算遵守无从判定，不计违规）",
        )
    cheapest = min(catalog_prices)
    if cheapest > max_price:
        return CheckResult.of(
            "prices_within", False,
            f"提及的目录价最低 {cheapest}，仍超预算 {max_price}",
        )
    return CheckResult.of(
        "prices_within", True,
        f"预算内选项已给出（最低目录价 {cheapest} ≤ {max_price}；"
        f"非目录价数字（差价/回显等）不判违规）",
    )


def _contains_pair(text: str, product_id: str, price: int) -> bool:
    """ID 或商品名出现其一 + 精确目录价出现（模型可能用名称而非 ID 指代）。"""
    text = _normalize(text)
    name = _PRODUCT_BY_ID.get(product_id, {}).get("name", "")
    identified = product_id in text or (name and name in text)
    return identified and str(price) in text


def check_must_include(case: EvalCase, trace: RunTrace) -> CheckResult:
    """精确价断言：每个 (ID, 目录价) 对都必须出现在回复中。"""
    spec = case.must_include_catalog_price
    if not spec:
        return CheckResult.skip("must_include")
    text = trace.all_replies_text
    missing = [
        f"{pid}(应含 {price})"
        for pid, price in spec.items()
        if not _contains_pair(text, pid, price)
    ]
    if missing:
        return CheckResult.of("must_include", False, f"缺失：{'、'.join(missing)}")
    return CheckResult.of("must_include", True, f"全部命中：{spec}")


def check_must_include_any(case: EvalCase, trace: RunTrace) -> CheckResult:
    """精确价断言（宽松版）：命中任意一对 (ID, 目录价) 即可。"""
    spec = case.must_include_catalog_price_any
    if not spec:
        return CheckResult.skip("must_include_any")
    text = trace.all_replies_text
    for pid, price in spec.items():
        if _contains_pair(text, pid, price):
            return CheckResult.of("must_include_any", True, f"命中 {pid}={price}")
    return CheckResult.of(
        "must_include_any", False,
        f"未命中任何一对：{spec}",
    )


def check_prices_in_catalog(case: EvalCase, trace: RunTrace) -> CheckResult:
    """回复中的价格必须都存在于任意品类目录（无编造价格）。"""
    if not case.reply_prices_in_catalog:
        return CheckResult.skip("prices_in_catalog")
    all_prices = set().union(*_CATEGORY_PRICES.values())
    # 用户输入里出现过的数字（预算回显）不算编造价格
    user_text = _normalize("\n".join(trace.turns))
    # 剔除建议性数字：区间（¥700-1000）与约数（¥1000 左右）——都是给用户的预算建议，不是商品价
    advisory_text = re.sub(r"\d{3,6}\s*[-—~至到]\s*\d{3,6}", "", trace.all_replies_text)
    advisory_text = re.sub(r"\d{3,6}\s*(?:左右|上下|附近)", "", advisory_text)
    fabricated = [
        v for v in _extract_prices(_normalize(advisory_text))
        if v not in all_prices and str(v) not in user_text
    ]
    if fabricated:
        return CheckResult.of("prices_in_catalog", False, f"编造的价格：{fabricated}")
    return CheckResult.of("prices_in_catalog", True, "提及的价格均真实")


def check_keywords(case: EvalCase, trace: RunTrace) -> CheckResult:
    """最终回复应命中全部关键词。"""
    if not case.keyword_hits:
        return CheckResult.skip("keywords")
    final = _normalize(trace.replies[-1] if trace.replies else "")
    missing = [kw for kw in case.keyword_hits if kw not in final]
    if missing:
        return CheckResult.of("keywords", False, f"未命中：{missing}")
    return CheckResult.of("keywords", True, f"命中：{case.keyword_hits}")


def check_min_chars(case: EvalCase, trace: RunTrace) -> CheckResult:
    """最终回复最低篇幅。"""
    if case.min_chars is None:
        return CheckResult.skip("min_chars")
    final = trace.replies[-1] if trace.replies else ""
    if len(final) < case.min_chars:
        return CheckResult.of(
            "min_chars", False,
            f"回复 {len(final)} 字 < 要求 {case.min_chars}",
        )
    return CheckResult.of("min_chars", True, f"回复 {len(final)} 字")


# ---------- 汇总入口 ----------

_CHECKS = (
    check_route,
    check_tools,
    check_valid_ids,
    check_prices_within,
    check_must_include,
    check_must_include_any,
    check_prices_in_catalog,
    check_keywords,
    check_min_chars,
)


def run_checks(case: EvalCase, trace: RunTrace) -> list[CheckResult]:
    """对一条轨迹应用用例声明的全部断言，返回逐条判定结果。"""
    return [check(case, trace) for check in _CHECKS]
