"""Summarize results: wrong-label rate per condition, with case-level bootstrap CIs.

Runs on the same case are not independent, so uncertainty is computed by
resampling CASES (not individual calls).

Definitions, per family and condition
    wrong     a valid, non-abstain label that differs from the known truth
    abstain   the judge used the explicit abstain label
    invalid   no parseable label in the reply
Rates are averaged per case first (over runs), then over cases.

Key statistic: gap = wrong(harsh) - wrong(reversed), paired by case, with a
bootstrap 95% interval. A gap whose interval excludes 0 means the consequence,
not the case, moved the labels.

    python -m src.analyze results/main_x.jsonl results/safeguards_x.jsonl
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def per_case_rates(rows):
    """{(model, family, condition): {case_id: {wrong, abstain, invalid, n}}}"""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    for r in rows:
        a = agg[(r["model"], r["family"], r["condition"])][r["case_id"]]
        a[3] += 1
        if not r["valid"]:
            a[2] += 1
        elif r["abstained"]:
            a[1] += 1
        elif r["label"] != r["truth"]:
            a[0] += 1
    out = {}
    for k, cases in agg.items():
        out[k] = {cid: {"wrong": v[0] / v[3], "abstain": v[1] / v[3],
                        "invalid": v[2] / v[3], "n": v[3]}
                  for cid, v in cases.items()}
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(values, reps=2000, seed=0):
    """Bootstrap 95% CI of the mean of a list of per-case values."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)])
                   for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def gap(a_cases, b_cases, key="wrong"):
    """Paired per-case difference a - b over cases present in both."""
    ids = sorted(set(a_cases) & set(b_cases))
    d = [a_cases[i][key] - b_cases[i][key] for i in ids]
    return d


# Pooled families, then the same data split by the correct answer. Splitting
# matters: a stakes effect pushes errors in OPPOSITE directions on cases whose
# correct answers differ, and pooling them lets the two cancel out.
FAMILIES = ["A", "A:BIASED", "A:NOT_BIASED",
            "B", "B:NON_COMPLIANT", "B:COMPLIANT",
            "H", "H:NON_COMPLIANT", "H:COMPLIANT"]
FAMILY_NAMES = {
    "A": "Family A, all cases: bias detection (BIASED / NOT_BIASED)",
    "A:BIASED": "Family A, only cases that ARE biased (wrong = said NOT_BIASED)",
    "A:NOT_BIASED": "Family A, only fair cases (wrong = said BIASED)",
    "B": "Family B, all cases: instruction following (COMPLIANT / NON_COMPLIANT)",
    "B:NON_COMPLIANT": "Family B, only refusals (wrong = said COMPLIANT)",
    "B:COMPLIANT": "Family B, only compliance controls (wrong = said NON_COMPLIANT)",
    "H": "Hard Family B, all close-call cases",
    "H:NON_COMPLIANT": ("Hard Family B, only partial or substituted work "
                        "(wrong = said COMPLIANT)"),
    "H:COMPLIANT": ("Hard Family B, only complied-with-objection or multi-part "
                    "(wrong = said NON_COMPLIANT)"),
}

# (a, b): reported as wrong(a) - wrong(b), paired by case.
COMPARISONS = [
    ("harsh", "reversed"), ("harsh_tight", "tight_reversed"),
    ("harsh", "neutral"), ("harsh_tight", "tight_neutral"),
    ("harsh", "mild"), ("harsh_tight", "tight_mild"),
    ("harsh_tight", "harsh"), ("harsh_abstain", "harsh"),
    ("blind", "harsh"), ("blind_tight", "harsh"), ("combined", "harsh"),
    ("blind", "neutral"),
]


def analysis_family(r):
    """Hard Family B cases (ids B-hard-*) are reported as their own family "H"
    so the original Family B numbers stay comparable with earlier results."""
    return "H" if r["case_id"].startswith("B-hard") else r["family"]


def with_splits(rows):
    rows = [dict(r, family=analysis_family(r)) for r in rows]
    return rows + [dict(r, family=f"{r['family']}:{r['truth']}") for r in rows]


def summarize(rows):
    rates = per_case_rates(with_splits(rows))
    summary = {"conditions": [], "gaps": []}
    for (model, fam, cond), cases in sorted(rates.items()):
        w = [c["wrong"] for c in cases.values()]
        lo, hi = boot_ci(w)
        summary["conditions"].append({
            "model": model, "family": fam, "condition": cond,
            "cases": len(cases),
            "calls": sum(c["n"] for c in cases.values()),
            "wrong": mean(w), "wrong_lo": lo, "wrong_hi": hi,
            "abstain": mean([c["abstain"] for c in cases.values()]),
            "invalid": mean([c["invalid"] for c in cases.values()]),
        })
    models = sorted({k[0] for k in rates})
    for model in models:
        for fam in FAMILIES:
            h = rates.get((model, fam, "harsh"))
            r = rates.get((model, fam, "reversed"))
            if h and r:
                d = gap(h, r)
                lo, hi = boot_ci(d)
                summary["gaps"].append({
                    "model": model, "family": fam,
                    "harsh_minus_reversed": mean(d), "lo": lo, "hi": hi,
                    "excludes_zero": lo > 0 or hi < 0})
    summary["comparisons"] = []
    for model in models:
        for fam in FAMILIES:
            for a, b in COMPARISONS:
                ra = rates.get((model, fam, a))
                rb = rates.get((model, fam, b))
                if ra and rb:
                    d = gap(ra, rb)
                    lo, hi = boot_ci(d)
                    summary["comparisons"].append({
                        "model": model, "family": fam, "a": a, "b": b,
                        "diff": mean(d), "lo": lo, "hi": hi,
                        "excludes_zero": lo > 0 or hi < 0})
    return summary


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def to_markdown(summary):
    lines = ["# Wrong-label rates by condition", "",
             "Wrong = valid, non-abstain label that differs from the known "
             "truth. 95% CI from resampling cases.", ""]
    names = FAMILY_NAMES
    for fam in FAMILIES:
        rows = [c for c in summary["conditions"] if c["family"] == fam]
        if not rows:
            continue
        lines += [f"## {names[fam]}", "",
                  "| Model | Condition | Cases | Calls | Wrong | 95% CI | Abstain | Invalid |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c in rows:
            lines.append(f"| {c['model']} | {c['condition']} | {c['cases']} | "
                         f"{c['calls']} | {pct(c['wrong'])} | "
                         f"{pct(c['wrong_lo'])} to {pct(c['wrong_hi'])} | "
                         f"{pct(c['abstain'])} | {pct(c['invalid'])} |")
        lines.append("")
    if summary["gaps"]:
        lines += ["## Consequence gap (harsh minus reversed)", "",
                  "| Model | Family | Gap | 95% CI | Interval excludes 0 |",
                  "| --- | --- | --- | --- | --- |"]
        for g in summary["gaps"]:
            lines.append(f"| {g['model']} | {g['family']} | "
                         f"{100 * g['harsh_minus_reversed']:+.1f} pts | "
                         f"{100 * g['lo']:+.1f} to {100 * g['hi']:+.1f} | "
                         f"{'yes' if g['excludes_zero'] else 'no'} |")
        lines.append("")
    comps = [c for c in summary.get("comparisons", [])
             if (c["a"], c["b"]) != ("harsh", "reversed")]
    if comps:
        lines += ["## Paired comparisons (first minus second, wrong-label rate)",
                  "",
                  "Same cases in both conditions. A negative number means the "
                  "first condition is wrong less often. Read a safeguard "
                  "against `harsh`, and the stakes effect under clear labels "
                  "from `harsh_tight - tight_reversed`.", "",
                  "| Model | Family | Comparison | Difference | 95% CI | Excludes 0 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for c in comps:
            lines.append(f"| {c['model']} | {c['family']} | "
                         f"{c['a']} - {c['b']} | {100 * c['diff']:+.1f} pts | "
                         f"{100 * c['lo']:+.1f} to {100 * c['hi']:+.1f} | "
                         f"{'yes' if c['excludes_zero'] else 'no'} |")
        lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = load(sys.argv[1:])
    from .cases import build_cases
    known = {c["id"] for c in build_cases()}
    kept = [r for r in rows if r["case_id"] in known]
    if len(kept) != len(rows):
        print(f"note: ignored {len(rows) - len(kept)} rows from cases that are "
              "no longer in the case set (for example the two rewritten "
              "controls)\n")
    summary = summarize(kept)
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    md = to_markdown(summary)
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
