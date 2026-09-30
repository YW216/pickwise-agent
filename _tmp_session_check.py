import json
import sys

sys.path.insert(0, ".")

from app.agent.context_budget import estimate_text_tokens
from app.config.settings import settings

out = []
out.append(
    f"[上限] summary_max_tokens={settings.summary_max_tokens} "
    f"summary_max_chars={settings.summary_max_chars} "
    f"keep_recent={settings.keep_recent_tokens}"
)
out.append("")

for name in ["session.json", "_compact_verify.json"]:
    path = f"app/sessions/{name}"
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        out.append(f"{name}: （不存在）")
        continue
    except Exception as e:
        out.append(f"{name}: 读取失败 {type(e).__name__}: {e}")
        continue

    summary = d.get("summary") or ""
    msgs = d.get("messages") or []
    prods = d.get("products") or {}
    out.append(f"{name}")
    out.append(f"  version = {d.get('version')}")
    out.append(f"  summary = {len(summary)} 字 / 约 {estimate_text_tokens(summary)} token")
    out.append(f"  messages = {len(msgs)} 条")
    out.append(f"  products = {len(prods)} 款")
    if summary:
        out.append("  --- summary 前 200 字 ---")
        out.append("  " + summary[:200].replace("\n", "\n  "))
    out.append("")

with open("_tmp_session_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("ok")
