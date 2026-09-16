"""临时验证：per-agent skill catalog 过滤（跑完可删）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.multi_agent.orchestrator import MultiAgentOrchestrator

orch = MultiAgentOrchestrator()
for key in ("guide", "compare", "consult"):
    cat = orch._skill_catalog_for(key)
    has_pr = "product-recommend" in cat
    poisoned = ("process-return" in cat) or ("track-order" in cat)
    print(f"{key:8s} catalog: {len(cat):4d} 字 | 含 product-recommend: {has_pr} | 客服毒化: {poisoned}")
