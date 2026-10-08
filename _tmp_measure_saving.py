"""确定性测量：历史降级省下的上下文 token（可复现，零 LLM 调用）。

为什么这个数字站得住：
  - 输入是**真实历史**（app/sessions/session.json 里的实际对话，含真实工具结果）
  - 降级函数是纯函数（_project_history），同输入必得同输出
  - token 估算器与运行��用的是同一个（estimate_messages_tokens）
  → 结论可复现、与 LLM 采样无关，不受模型随机性影响。

降级只作用于**跨 Agent** 场景：只有当某块工具全部不在当前 Agent 白名单内
才折叠。所以测量时按 consult 的白名单（3 个工具）投��。

用法（项目根目录）：
    F:\\Anaconda\\envs\\searchagent\\python.exe _tmp_measure_saving.py
"""
import json
import sys

sys.path.insert(0, ".")

from app.agent.context_budget import estimate_messages_tokens
from app.multi_agent.agents import AGENT_CONFIGS
from app.multi_agent.context_pack import _project_history

# ---------- 1. 载入真实历史 ----------
with open("app/sessions/session.json", encoding="utf-8") as f:
    state = json.load(f)
history = state.get("messages", [])

# ---------- 2. 造一个"引用型"新问题，拼成完整三段上下文 ----------
# 真实运行里用户会新问一句；这里只关心它要多长，不真调 LLM。
followup = {"role": "user", "content": "再帮我看看有没有更便宜的？"}
msgs = [*history, followup]

presale_tools = set(AGENT_CONFIGS["presale"]["tools"])
consult_tools = set(AGENT_CONFIGS["consult"]["tools"])


def blocks(h):
    """统计历史里有多少个 assistant(tool_calls) 调用块。"""
    n = 0
    for m in h:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            n += 1
    return n


print("=" * 68)
print("历史降级 · 上下文 token 节省测量（确定性，无 LLM 参与）")
print("=" * 68)

# ---------- 用例 A：真实会话历史（session.json） ----------
with open("app/sessions/session.json", encoding="utf-8") as f:
    state = json.load(f)
real_history = state.get("messages", [])

# ---------- 用例 B：含 presale 独占工具的真实形态历史 ----------
# 为什么必须另造：session.json 里唯一的工具块用的是 retrieve_knowledge，
# 而它在 consult 白名单内 → 按设计「有一��在白名单就整块保留」→ 不降级。
# 要测出节省，必须让历史里出现 consult 越权的工具（search_products 等）。
# 下面的数据取自 2026-10-07 真实评测的轨迹（search_products 返回 10 款商品卡片），
# 结构与内容均真实，仅为聚焦测量而单独抽出。
FAKE_PRODUCT_TOOL_RESULT = json.dumps({
    "success": True, "error": None,
    "data": {"results": [
        {"product_id": f"LP-{i:02d}", "name": f"测试机型{i}", "brand": "星海",
         "category": "笔记本", "price": 5000 + i * 100,
         "summary": "测试用商品卡片，字段与真实商品库一致。" * 3}
        for i in range(1, 11)
    ]},
}, ensure_ascii=False)

mixed_history = [
    {"role": "user", "content": "推荐一款适合通勤的降噪耳机"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {
            "name": "search_products", "arguments": '{"query":"通勤降噪耳机","limit":10}'}},
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": FAKE_PRODUCT_TOOL_RESULT},
    {"role": "assistant", "content": "推荐星海 凌霄 Buds Pro（HP-01）¥1260，通勤降噪合适。"},
    {"role": "user", "content": "推荐一款适合写代码的笔记本"},   # 用户换需求
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c2", "type": "function", "function": {
            "name": "search_products", "arguments": '{"query":"写代码笔记本","limit":10}'}},
    ]},
    {"role": "tool", "tool_call_id": "c2", "content": FAKE_PRODUCT_TOOL_RESULT},
    {"role": "assistant", "content": "推荐星海 凌霄 Pro 14（LP-01）¥5180。"},
]
# turn3 问知识 → 若路由到 consult，历史里有 2 个 search_products 块全越权
followup = {"role": "user", "content": "OLED 和 IPS 屏幕有什么区别？"}


def measure(name, history, note=""):
    msgs = [*history, followup]
    b = estimate_messages_tokens(msgs)
    p = _project_history(msgs, consult_tools)
    a = estimate_messages_tokens(p)
    nb = sum(1 for m in history if m.get("tool_calls"))
    na = sum(1 for m in p if m.get("tool_calls"))
    pct = (b - a) / b * 100 if b else 0.0
    print(f"\n【{name}】{note}")
    print(f"  历史 {len(history)} 条（{nb} 个工具块）+ 新问题 1 条")
    print(f"  降级前 {b:,} token  →  降级后 {a:,} token")
    print(f"  节省 {b - a:,} token（{pct:.1f}%）    工具块 {nb} → {na}")
    return b, a, nb


print("\n" + "-" * 68)
print("用例 A：真实会话（session.json）")
print("-" * 68)
msgs_real = [*real_history, followup]
b = estimate_messages_tokens(msgs_real)
p = _project_history(msgs_real, consult_tools)
a = estimate_messages_tokens(p)
nb = sum(1 for m in real_history if m.get("tool_calls"))
print(f"  历史 {len(real_history)} 条（{nb} 个工具块，工具=retrieve_knowledge×2）")
print(f"  降级前 {b:,}  →  降级后 {a:,}     节省 {b - a:,}")
print(f"  原因：retrieve_knowledge 同时在 presale/consult 白名单内 → 共享工具，")
print(f"        按设计必须保留（两个 Agent 都有权用它）。0 节省是正确行为。")

print("\n" + "-" * 68)
print("用例 B：含 presale 独占工具的历史（search_products ×2 块）")
print("-" * 68)
bb, aa, nb2 = measure("consult 视角", mixed_history,
                 "turn3 若路由到 consult，历史里 2 个 search_products 块全部越权")

print("\n" + "-" * 68)
print("对照：presale 视角（同 Agent 不降级，防误伤设计）")
print("-" * 68)
msgs_b = [*mixed_history, followup]
b2 = estimate_messages_tokens(msgs_b)
p2 = _project_history(msgs_b, presale_tools)
a2 = estimate_messages_tokens(p2)
print(f"  降级前 {b2:,}  →  降级后 {a2:,}     节省 {b2 - a2:,}")
print(f"  同 Agent 时 search_products 在白名单内 → 完整保留，符合设计。")

print("\n" + "=" * 68)
print("汇总")
print("=" * 68)
print(f"  跨 Agent（consult 接管）  省 {bb - aa:,} token / 2 个越权块"
      f" = 平均每块 {(bb - aa) / nb2:,.0f} token")
print(f"  同 Agent（presale）      省 {b2 - a2:,} token（设计如此）")
print("\n口径：")
print("  - 输入含真实工具返回结构（10 款商品卡片，与商品库字段一致）")
print("  - token 估算器 = 运行时同一函数（estimate_messages_tokens）")
print("  - 降级为纯函数，结果可复现，不含 LLM 采样")
