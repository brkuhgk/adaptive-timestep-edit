"""Stage 7: fill report/report.md and report/email_template.txt from work/results/summary.json.

Writes report/report_filled.md and report/email_filled.txt. Unknown placeholders become [n/a].
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from ate.cli import base_parser, setup  # noqa: E402


def derived(summary: dict, results: Path) -> dict:
    v = dict(summary)
    ci = summary.get("best_practical_dJ_ci") or [None, None, None]
    v["best_practical_dJ"], v["best_practical_dJ_lo"], v["best_practical_dJ_hi"] = ci
    for key in ("lowest_t_category", "highest_t_category"):
        if summary.get(key):
            v[f"{key}_name"], v[f"{key}_t"] = summary[key]
    if summary.get("oracle_in_sample_J") is not None and summary.get("reference_J") is not None:
        v["oracle_gap"] = summary["oracle_in_sample_J"] - summary["reference_J"]
    name = str(summary.get("best_practical_method", ""))
    v["best_practical_method_short"] = (
        "a training-free probe of where the two prompts' predictions diverge" if name.startswith("Velocity")
        else "a small LLM that reads the two prompts" if name.startswith("LLM")
        else "a keyword rule over the two prompts" if name.startswith("Prompt keywords") else name)
    probe = results / "probe_long.csv"
    if probe.exists():
        v["probe_unet_calls"] = float(pd.read_csv(probe)["unet_calls"].iloc[0])
    table = results / "main_table.md"
    v["MAIN_TABLE"] = table.read_text().strip() if table.exists() else "[run scripts/06_analyze.py]"
    return v


def fill(text: str, values: dict) -> str:
    def repl(m: re.Match) -> str:
        key, _, fmt = m.group(1).partition(":")
        val = values.get(key)
        if val is None or (isinstance(val, float) and val != val):
            return "[n/a]"
        try:
            return format(val, fmt) if fmt else str(val)
        except (TypeError, ValueError):
            return str(val)
    return re.sub(r"\{\{([^}]+)\}\}", repl, text)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    settings, _ = setup(args)
    res = settings.results
    summary = json.loads((res / "summary.json").read_text())
    values = derived(summary, res)
    rep = Path(__file__).resolve().parent.parent / "report"
    (rep / "report_filled.md").write_text(fill((rep / "report.md").read_text(), values))
    (rep / "email_filled.txt").write_text(fill((rep / "email_template.txt").read_text(), values))
    print(f"Wrote {rep / 'report_filled.md'} and {rep / 'email_filled.txt'}")


if __name__ == "__main__":
    main()
