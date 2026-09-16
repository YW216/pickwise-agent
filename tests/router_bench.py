"""Router 路由准确率评测（真调 LLM）。

评测目标：量化当前 Router 在不同场景下的分类准确率，定位"路由漂移"的重灾区。
评测维度：
- single_presale / single_consult：单意图独立问题（无历史）
- multi：多意图（期望多值输出）
- continue_*：多轮澄清延续（带 history，短回答）
- switch_*：多轮场景切换（带 history，新问题长句）

预期判定依据：ROUTER_PROMPT 规则 4/5（知识/单款凭据分流）与规则 6（多轮延续）。
（2026-09-15 随架构 PickWise 化更新：guide/compare 场景收敛为 presale）
注意：商品名使用虚构品牌（星海/曜石/云章/墨白/极光），避免模型因认识商品而影响分类。

用法：python tests/router_bench.py [--cases 指定分组]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.multi_agent.router import Router  # noqa: E402


# 澄清中的历史（presale 澄清询问预算/用途）
H_GUIDE_CLARIFYING = [
    {"role": "user", "content": "我想买个耳机"},
    {"role": "assistant",
     "content": "好的～先确认两个关键信息：1. 预算大概多少？2. 主要用来做什么（通勤/运动/打游戏/办公）？"},
]

# 咨询后的历史（consult 回答了 OLED 知识）
H_CONSULT_ANSWERED = [
    {"role": "user", "content": "OLED 屏伤眼吗？"},
    {"role": "assistant",
     "content": "OLED 采用 PWM 调光，低频（约 240Hz）下敏感人群可能眼疲劳，建议选高频 PWM 或 DC 调光机型。"},
]

# 推荐后的历史（presale 给了推荐）
H_GUIDE_RECOMMENDED = [
    {"role": "user", "content": "推荐个 6000 左右的办公笔记本"},
    {"role": "assistant",
     "content": "推荐星海凌霄 Pro 14（¥5180）：1.32kg、OLED 屏、续航 11 小时，办公很适合。"},
]

# 真实失败场景复现（用户 2026-09-01 实测："600" 被路由到 consult 时的完整上下文）：
# 第 1 轮：用户买耳机 → presale 澄清（纯提问）
# 第 2 轮：用户答 500 → presale 搜空后回复（"建议放宽 + 问场景"的混合消息）
H_600_REAL_FAIL = [
    {"role": "user", "content": "我想买副耳机"},
    {"role": "assistant",
     "content": "好的，选耳机这事咱们来理一理～先问两个最关键的：1. 预算大概多少？2. 主要在哪用？比如通勤/地铁（优先降噪）、办公室/自习（优先佩戴舒适）、运动跑步（优先防水）、打游戏/影音（优先音质、延迟）？"},
    {"role": "user", "content": "500"},
    {"role": "assistant",
     "content": "我查了一下目录，500 元以内的耳机暂时没有匹配的商品。给你两个下一步选择：1. 预算稍微放宽一点——比如告诉我 600 或 800 以内，我帮你看看有没有合适的；2. 如果预算能到千元档，候选会丰富很多。另外想确认一下：你平时主要在什么场景用耳机？这会影响推荐方向（比如通勤要降噪、运动要防水）。"},
]


CASES: list[dict] = [
    # ---------- 单意图：guide ----------
    {"group": "single_presale", "input": "有什么适合写代码的笔记本推荐吗", "history": None,
     "expected": ["presale"], "note": "求推荐"},
    {"group": "single_presale", "input": "预算6000左右买什么手机", "history": None,
     "expected": ["presale"], "note": "求推荐+预算"},
    {"group": "single_presale", "input": "从我的收藏里挑一款耳机", "history": None,
     "expected": ["presale"], "note": "收藏挑选"},
    {"group": "single_presale", "input": "我想买个轻薄本", "history": None,
     "expected": ["presale"], "note": "明确购买意图"},

    # ---------- 单意图：compare ----------
    {"group": "single_presale", "input": "星海凌霄 Pro 14 和曜石磐石 15 哪个好", "history": None,
     "expected": ["presale"], "note": "两款对比（求推荐与求对比同属售前）"},
    {"group": "single_presale", "input": "云章轻羽 Air 和墨白素笺 Earbuds 怎么选", "history": None,
     "expected": ["presale"], "note": "怎么选"},
    {"group": "single_presale", "input": "这两款差在哪", "history": None,
     "expected": ["presale"], "note": "差在哪（指代依赖历史，无历史属理想情况）"},

    # ---------- 单意图：consult ----------
    {"group": "single_consult", "input": "OLED 和 IPS 屏有什么区别", "history": None,
     "expected": ["consult"], "note": "知识问题"},
    {"group": "single_consult", "input": "耳机一般保修几年", "history": None,
     "expected": ["consult"], "note": "政策问题"},
    {"group": "single_consult", "input": "你好", "history": None,
     "expected": ["consult"], "note": "闲聊兜底"},

    # ---------- 多意图 ----------
    {"group": "multi", "input": "推荐个笔记本，顺便讲讲 OLED 和 IPS 的区别", "history": None,
     "expected": ["presale", "consult"], "note": "推荐+知识"},
    {"group": "multi", "input": "帮我挑个耳机，顺便对比下凌霄 Buds Pro 和素笺 Earbuds", "history": None,
     "expected": ["presale"], "note": "推荐+对比（两诉求同属售前，收敛为单场景）"},

    # ---------- 澄清延续（短回答） ----------
    {"group": "continue", "input": "500", "history": H_GUIDE_CLARIFYING,
     "expected": ["presale"], "note": "补预算"},
    {"group": "continue", "input": "日常办公", "history": H_GUIDE_CLARIFYING,
     "expected": ["presale"], "note": "补用途"},
    {"group": "continue", "input": "打游戏", "history": H_GUIDE_CLARIFYING,
     "expected": ["presale"], "note": "补用途"},
    {"group": "continue", "input": "600", "history": H_600_REAL_FAIL,
     "expected": ["presale"], "note": "真实失败场景复现：两轮澄清+建议混合消息"},

    # ---------- 咨询追问（短回答） ----------
    {"group": "continue", "input": "那 IPS 呢", "history": H_CONSULT_ANSWERED,
     "expected": ["consult"], "note": "知识追问"},
    {"group": "continue", "input": "那续航呢", "history": H_GUIDE_RECOMMENDED,
     "expected": ["presale"], "note": "边界：推荐后追问参数 → 规则 6 参数/价格类延续归 presale"},

    # ---------- 场景切换（长句新问题） ----------
    {"group": "switch", "input": "OLED 和 IPS 屏有什么区别", "history": H_GUIDE_RECOMMENDED,
     "expected": ["consult"], "note": "推荐后问知识"},
    {"group": "switch", "input": "凌霄 Pro 14 和磐石 15 哪个好", "history": H_GUIDE_RECOMMENDED,
     "expected": ["presale"], "note": "推荐后问比选"},
    {"group": "switch", "input": "这款保修几年", "history": H_GUIDE_RECOMMENDED,
     "expected": ["consult"], "note": "推荐后问政策"},
]


def run() -> None:
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    router = Router(client, settings.model_name)

    groups = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    cases = CASES if "all" in groups else [c for c in CASES if c["group"] in groups]

    stats: dict[str, list[bool]] = {}
    errors: list[dict] = []

    print("=" * 70)
    print(f"Router 准确率评测（{len(cases)} 例，真调 {settings.model_name}）")
    print("=" * 70)

    for i, c in enumerate(cases, 1):
        try:
            actual = router.route(c["input"], c["history"])
        except Exception as e:
            actual = [f"ERROR:{e}"]
        ok = actual == c["expected"]
        stats.setdefault(c["group"], []).append(ok)
        mark = "✅" if ok else "❌"
        print(f"{mark} [{c['group']:14s}] {c['input']!r:40s} → {actual} (期望 {c['expected']})")
        if not ok:
            errors.append({**c, "actual": actual})

    print("\n" + "=" * 70)
    print("分场景准确率")
    print("=" * 70)
    total, total_ok = 0, 0
    for group, results in sorted(stats.items()):
        n, ok_n = len(results), sum(results)
        total += n
        total_ok += ok_n
        print(f"  {group:16s} {ok_n}/{n} = {ok_n / n:.0%}")
    print(f"  {'TOTAL':16s} {total_ok}/{total} = {total_ok / total:.0%}")

    if errors:
        print("\n错误用例明细：")
        for e in errors:
            print(f"  ❌ [{e['group']}] {e['note']}: {e['input']!r} → {e['actual']}（期望 {e['expected']}）")


if __name__ == "__main__":
    run()
