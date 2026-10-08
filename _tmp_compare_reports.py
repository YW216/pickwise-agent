"""评测报告对比：基线（降级前）vs 当前（降级后）。

用法（项目根目录）：
    F:\\Anaconda\\envs\\searchagent\\python.exe _tmp_compare_reports.py
"""
import json
import os

BASE = "app/evaluation/runs/report_BEFORE_degrade.json"   # 降级前（本次实测基线）
NOW = "app/evaluation/runs/report.json"                  # 降级后


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stat(d):
    s = d["summary"]
    errs = [c for c in d["cases"] if c.get("error")]
    tools = sum(len(c.get("checks") or []) for c in d["cases"])
    return {
        "用例数": s["total"],
        "通过": s["passed"],
        "异常用例": len(errs),
        "总token": s["total_tokens"],
    }


base, now = load(BASE), load(NOW)

print("=" * 62)
print("PickWise 端到端评测对比（降级前 / 降级后）")
print("=" * 62)

if base is None:
    print(f"[缺] 基线文件不存在: {BASE}")
else:
    print("【基线 = 2026-09-17 旧报告（仅 1 条用例，非本次跑出）】")
    for k, v in stat(base).items():
        print(f"  {k:10s} {v}")

if now is None:
    print(f"\n[缺] 当前报告不存在: {NOW}（评测是否已跑？）")
else:
    print("\n【当前 = 最近一次 run_eval 产出】")
    for k, v in stat(now).items():
        print(f"  {k:10s} {v}")

    print("\n--- 关键校验 ---")
    errs = [c["case_id"] for c in now["cases"] if c.get("error")]
    if errs:
        print(f"  [警告] 有 {len(errs)} 条用例报异常: {errs[:5]}")
        print("         若基线也异常 -> 插桩修复可能未生效，先别看 token 对比")
    else:
        print("  [OK] 无异常用例 —— sandbox 插桩已修好（error 全为 null）")

    if base and base["summary"]["total"] == now["summary"]["total"] and now["summary"]["total"] > 1:
        b, n = base["summary"]["total_tokens"], now["summary"]["total_tokens"]
        if b:
            print(f"\n  token: {b} -> {n}  ({(n - b) / b * 100:+.1f}%)")
    elif base:
        print("\n  [注意] 基线只有 1 条用例，与当前条数不同，token 不可直接比。")
        print("         要对比请先 git stash 掉 context_pack.py 重跑基线。")
