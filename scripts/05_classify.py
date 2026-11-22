"""Stage 5: predict each sample's edit type from its prompt pair (keyword rule + optional small LLM)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ate.cli import LOG, base_parser, setup  # noqa: E402
from ate.prompt_classifier import classify_all  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    settings, items = setup(args)
    ccfg = settings.raw["classifier"]
    use_llm = bool(ccfg.get("use_llm", True)) and not args.no_llm
    rows = classify_all(items, use_llm, ccfg["model"], int(ccfg.get("max_new_tokens", 4)))
    out = settings.results / "classifier.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    acc_kw = sum(r["keyword_category"] == r["true_category"] for r in rows) / len(rows)
    acc_llm = sum(r["llm_category"] == r["true_category"] for r in rows) / len(rows)
    LOG.info("Classifier accuracy: keyword %.1f%%, llm %.1f%% (llm used: %s). Wrote %s",
             100 * acc_kw, 100 * acc_llm, use_llm, out)


if __name__ == "__main__":
    main()
