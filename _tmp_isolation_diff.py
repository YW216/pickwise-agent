"""隔离回归对比：降级前 vs 降级后。

从 report.json 里抽出**被拒（失败）的工具调用**——这是判断"模型是否模仿
越权工具"的唯一硬证据。被拒 = 执行层按白名单挡下的越权调用。

用法（项目根目录，PowerShell）：
    F:\\Anaconda\\envs\\searchagent\\python.exe _tmp_isolation_diff.py
"""
import json
import sys

sys.path.insert(0, ".")

BEFORE = "app/evaluation/runs/report_BEFORE_degrade.json"
AFTER = "app/evaluation/runs/report.json"


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def rejected_of(case):
    """从 checks 结果里找失败项——失败即工具被拒（success=false）。"""
    out = []
    for ck in case.get("checks") or []:
        if ck["name"] == "tools" and not ck["passed"]:
            out.append(ck.get("detail", ""))
    return out


def main():
    b, a = load(BEFORE), load(AFTER)
    if not b or not a:
        print("缺报告文件：需先各跑一次 run_eval --dataset cases_isolation.json")
        return

    def index(d):
        return {c["case_id"]: c for c in d["cases"]}

    bi, ai = index(b), index(a)

    print("=" * 72)
    print("隔离回归对比（降级前 / 降级后）")
    print("=" * 72)
    print(f"{'用例':32s} {'前token':>9s} {'后token':>9s} {'前失败':>7s} {'后失败':>7s}")
    print("-" * 72)

    tot_b = tot_a = 0
    worse = []
    for cid in bi:
        cb, ca = bi[cid], ai.get(cid)
        if not ca:
            print(f"{cid:32s} {'（降级后缺此用例）'}")
            continue
        rb, ra = rejected_of(cb), rejected_of(ca)
        tb = cb.get("checks") and 0 or 0
        tok_b = tok_a = None
        # report.json 的每条用例不带 token，token 在 summary；此处只比失败项
        mark = ""
        if rb and not ra:
            mark = "  <== 越权被拒消失 ✅"
        elif rb and ra:
            mark = "  （两侧都拒）"
        elif not rb and ra:
            mark = "  [注意] 降级后新出现失败"
            worse.append(cid)
        print(f"{cid:32s} {str(tok_b or '-'):>9s} {str(tok_a or '-'):>9s} "
              f"{len(rb):>7d} {len(ra):>7d}{mark}")

    print("-" * 72)
    for label, d in (("降级前", b), ("降级后", a)):
        s = d["summary"]
        errs = sum(1 for c in d["cases"] if c.get("error"))
        print(f"{label}: 用例 {s['total']} | 通过 {s['passed']} | 异常 {errs} | token {s['total_tokens']}")
        tot_b += 0
    if b["summary"]["total_tokens"]:
        old, new = b["summary"]["total_tokens"], a["summary"]["total_tokens"]
        print(f"\ntoken: {old} -> {new}  ({(new - old) / old * 100:+.1f}%)")

    if worse:
        print("\n[回归警告] 降级后出现新的失败用例: " + ", ".join(worse))
    print("\n判读：'前失败' 降到 0 且 '后失败' 为 0 → 越权模仿被消除")


if __name__ == "__main__":
    main()
