"""Agent 评估模块（P6）。

端到端评估闭环：测试用例管理（PickWise 语义）、沙箱采集（隔离 + 插桩）、
规则判分（确定性断言，ground truth = 本地 mock 目录）、聚合报告。

核心组件：
- Sandbox：隔离、可复现地重跑测试集，并采集运行全过程（token / 路由 / 工具轨迹 / 回复）。
- Evaluator：在采集到的轨迹上应用 metrics 的规则断言，聚合为分类别通过率。
- metrics：规则判分库（价格 ∈ 目录、ID 不编造、关键词命中），全部确定性、零 LLM。
"""

from app.evaluation.dataset import EvalCase, load_dataset
from app.evaluation.evaluator import CaseResult, EvalReport, Evaluator
from app.evaluation.sandbox import Sandbox
from app.evaluation.trace import LLMCallRecord, RunTrace, ToolObservation

__all__ = [
    "EvalCase",
    "load_dataset",
    "Sandbox",
    "RunTrace",
    "LLMCallRecord",
    "ToolObservation",
    "Evaluator",
    "CaseResult",
    "EvalReport",
]
