#!/usr/bin/env python3
"""Fill in the missing half of REPORT.md §6.

results/qualitative.json (written by NB5) only stores the fine-tune's prediction per
item -- never baseline (b)'s, so REPORT.md's qualitative table cannot be filled in
honestly from results/ alone. This re-runs ONLY baseline (b) (optimized prompt, base
model, no adapter) on the specific ticket indices NB5 flagged as imperfect for the
fine-tune, plus a couple of its clean wins, and prints a table with (b) and (c) side by
side so the report's "≥2 ca fine-tune THUA" requirement can be checked against real
data instead of guessed.

Needs a GPU (same tier as the rest of the lab) -- run this in the same Colab/Kaggle
session, AFTER nb2/nb5 have produced results/qualitative.json.

    python scripts/qualitative_baseline_b.py
    python scripts/qualitative_baseline_b.py --n-extra 4   # more clean-win rows
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-extra", type=int, default=2,
                     help="how many additional ft_score==1.0 rows to include for contrast")
    args = ap.parse_args()

    from labkit import config, evaluate, generate
    from labkit.config import get_tier

    qual = json.loads((ROOT / "results" / "qualitative.json").read_text(encoding="utf-8"))
    eval_rows = [json.loads(l) for l in
                 (ROOT / "data" / "eval_target.jsonl").read_text(encoding="utf-8").splitlines()]

    imperfect = [q for q in qual if q["ft_score"] < 1.0]
    perfect = [q for q in qual if q["ft_score"] >= 1.0][:args.n_extra]
    chosen = imperfect + perfect
    if not chosen:
        print("nothing to compare — every item scored 1.0 and --n-extra is 0", file=sys.stderr)
        return 1

    # qualitative.json's "ticket" field is eval_target's `input` truncated to 70 chars.
    by_prefix = {r["input"][:70]: r for r in eval_rows}
    picked = [(q, by_prefix[q["ticket"]]) for q in chosen if q["ticket"] in by_prefix]

    tier = get_tier()
    model, tok = generate.load_base(tier)
    prompts = [r["input"] for _, r in picked]
    completions, _ = generate.generate_batch(
        model, tok, prompts, system=config.OPTIMIZED_PROMPT, label="qualitative (b)")

    rows = []
    for (q, r), b_pred in zip(picked, completions):
        b_score = evaluate.triage_field_accuracy(b_pred, r["label"])
        rows.append({
            "ticket": q["ticket"],
            "gold": r["label"],
            "b_pred": b_pred.replace("\n", " ")[:120],
            "b_score": round(b_score, 2),
            "ft_pred": q["ft_pred"],
            "ft_score": q["ft_score"],
        })

    out = ROOT / "results" / "qualitative_with_b.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'ticket':<45} {'gold.intent':<14} {'b':>5} {'c':>5}  verdict")
    print("-" * 90)
    for row in rows:
        verdict = ("FT THUA" if row["ft_score"] < row["b_score"] else
                    "FT THANG" if row["ft_score"] > row["b_score"] else "HOA")
        print(f"{row['ticket'][:44]:<45} {row['gold']['intent']:<14} "
              f"{row['b_score']:>5.2f} {row['ft_score']:>5.2f}  {verdict}")
    print(f"\nwrote {out} -- paste rows into REPORT.md §6 "
          "(need >=2 rows marked FT THUA to satisfy rubric 3.4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
