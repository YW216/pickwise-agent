"""评估执行器：编排 沙箱跑用例 → 规则判分 → 聚合报告（P6）。

Evaluator 自己不碰 Agent：让 Sandbox 跑出 RunTrace，再用 metrics 的规则断言
逐条判定（全部确定性规则，不调用 LLM judge——理由见 11-评估.md 第三节），
最后聚合为分类别通过率与总通过率。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from app.config.settings import settings
from app.evaluation import metrics, judges
from app.evaluation.dataset import EvalCase
from app.evaluation.reporter import LangfuseReporter
from app.evaluation.sandbox import Sandbox
from app.evaluation.trace import RunTrace


@dataclass
class CaseResult:
    """单条用例的评估结果：逐条断言判定 + 汇总。"""

    case_id: str
    category: str
    description: str
    checks: list = field(default_factory=list)  # metrics.CheckResult 列表
    judge_scores: dict = field(default_factory=dict)  # {aspect: {"score":1-5,"reasons":[...]}}；advisory 不参与 passed
    tool_hits: tuple | None = None  # (期望工具命中数, 期望总数)；未声明 expected_tools 为 None
    error: str | None = None  # 沙箱运行异常（非断言失败）
    trace: RunTrace | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        return all(c.passed for c in self.checks if c.applicable)

    @property
    def failures(self) -> list:
        return [c for c in self.checks if c.applicable and not c.passed]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "error": self.error,
            "checks": [
                {"name": c.name, "applicable": c.applicable,
                 "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
            "judge_scores": self.judge_scores,
            "tool_hits": list(self.tool_hits) if self.tool_hits else None,
        }


@dataclass
class EvalReport:
    """全量聚合：分分类通过率 + 总通过率 + token 汇总。"""

    results: list[CaseResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def category_summary(self) -> dict[str, tuple[int, int]]:
        """{分类: (通过数, 总数)}，保持用例文件中的出现顺序。"""
        summary: dict[str, tuple[int, int]] = {}
        for r in self.results:
            passed, total = summary.get(r.category, (0, 0))
            summary[r.category] = (passed + (1 if r.passed else 0), total + 1)
        return summary

    def to_dict(self) -> dict:
        return {
            "cases": [r.to_dict() for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": self.passed_count,
                "total_tokens": sum(
                    r.trace.total_tokens for r in self.results if r.trace
                ),
                "by_category": {
                    cat: {"passed": p, "total": t}
                    for cat, (p, t) in self.category_summary().items()
                },
            },
        }


class Evaluator:
    """逐用例评估：沙箱采集 → 规则判分 →（声明了 aspects 的用例）LLM judge。

    judge 分 advisory：写入 judge_scores，不参与 passed；解析失败标"未判"。
    判分完成后（可选）把结果作为 score 上报 Langfuse——判分与本地上报严格解耦：
    上报失败只记警告，不影响通过率与报告。
    """

    def __init__(self, sandbox: Sandbox, client: OpenAI | None = None,
                 model: str | None = None, reporter: LangfuseReporter | None = None):
        self.sandbox = sandbox
        self.client = client    # judge 用；None = 跳过全部 judge（离线测试）
        self.model = model or ""
        self.reporter = reporter or LangfuseReporter.disabled()

    def run_case(self, case: EvalCase) -> CaseResult:
        trace = self.sandbox.run(case)
        result = CaseResult(
            case_id=case.id,
            category=case.category,
            description=case.description,
            trace=trace,
            error=trace.error,
        )
        if not trace.error:
            result.checks = metrics.run_checks(case, trace)
            result.judge_scores = self._run_judges(case, trace)
            if case.expected_tools:
                succeeded = set(metrics.successful_tool_names(trace))
                result.tool_hits = (
                    len(set(case.expected_tools) & succeeded),
                    len(case.expected_tools),
                )
        # 上报判分结果（no-op 时零开销）
        self.reporter.report_case(
            trace.langfuse_trace_id,
            passed=result.passed,
            checks=result.checks,
            judge_scores=result.judge_scores,
            tool_hits=result.tool_hits,
            error=result.error,
        )
        return result

    def _run_judges(self, case: EvalCase, trace: RunTrace) -> dict:
        if not case.judge_aspects or self.client is None:
            return {}
        if not settings.eval_use_judge:
            return {}
        # judge 的 client 也会被 langfuse 的类级补丁接到（wrapt 全局补丁）——
        # 确认已包装才传 name（否则会漏给真 API）；同时把 judge 挂回用例 trace
        # （否则每次 judge 各成一个独立根 trace，看板全是噪音）
        instrumented = (
            self.reporter.enabled and self.reporter.is_instrumented(self.client)
        )
        scores = {}
        for aspect in case.judge_aspects:
            with self.reporter.judge_scope(trace.langfuse_trace_id, aspect):
                scores[aspect] = judges.judge_aspect(
                    client=self.client,
                    model=self.model,
                    aspect=aspect,
                    turns=self._judge_materials(trace),
                    call_name=f"judge:{aspect}" if instrumented else "",
                )
        return scores

    @staticmethod
    def _judge_materials(trace: RunTrace) -> list[dict]:
        """按轮组装 judge 材料：question / reply / 当轮工具调用三者对齐。

        v1 bug：只传最后一轮问题 + 全会话调用序列，multi-turn 用例第一轮的
        正当工作被判"与当前问题无关"。切分依据 sandbox 逐轮记录的
        tool_boundaries；若边界不完整（异常中途退出，此时 _run_judges 本不该
        被调到），兜底把全序列挂到最后一轮，宁可材料粗、不至错配。
        """
        bounds = trace.tool_boundaries
        aligned = len(bounds) == len(trace.turns)
        materials = []
        for i, question in enumerate(trace.turns):
            if aligned:
                start = bounds[i - 1] if i > 0 else 0
                obs = trace.tool_observations[start:bounds[i]]
            else:
                obs = trace.tool_observations if i == len(trace.turns) - 1 else []
            materials.append({
                "question": question,
                "reply": trace.replies[i] if i < len(trace.replies) else None,
                "tool_calls": [
                    {"name": o.name, "arguments": o.arguments} for o in obs
                ],
            })
        return materials

    def run_all(self, cases: list[EvalCase]) -> EvalReport:
        report = EvalReport()
        for case in cases:
            report.results.append(self.run_case(case))
        return report
