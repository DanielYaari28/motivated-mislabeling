"""Look inside the results: what did the judge actually say, and why?

    python -m src.explore results/main_mini.jsonl
    python -m src.explore results/main_mini.jsonl results/tight_mini.jsonl \
        --family A --condition harsh_tight --samples 3

Prints, for each (model, family, condition):
  * how often the judge gave each label (an "always says X" judge shows up here)
  * wrong-label rate split by the correct answer, i.e. the DIRECTION of errors
  * average output tokens (a large number means hidden reasoning is being billed)
Then the cases the judge got wrong most often, and sample replies for wrong
answers so you can read the judge's own explanation.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_wrong(r):
    return r["valid"] and not r["abstained"] and r["label"] != r["truth"]


def label_of(r):
    return r["label"] if r["valid"] else "(invalid)"


def summarize_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["family"], r["condition"])].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        labels = Counter(label_of(r) for r in rs)
        by_truth = defaultdict(lambda: [0, 0])      # truth -> [wrong, n]
        for r in rs:
            by_truth[r["truth"]][1] += 1
            by_truth[r["truth"]][0] += is_wrong(r)
        out.append({
            "key": key, "n": len(rs),
            "labels": {k: v / len(rs) for k, v in labels.items()},
            "wrong": sum(is_wrong(r) for r in rs) / len(rs),
            "by_truth": {t: (w / n if n else float("nan"), n)
                         for t, (w, n) in by_truth.items()},
            "avg_out_tokens": sum(r.get("out_tokens", 0) for r in rs) / len(rs),
        })
    return out


def case_rates(rows):
    """{(model, condition): [(case_id, truth, wrong_rate, n)]} worst first."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, None]))
    for r in rows:
        a = agg[(r["model"], r["condition"])][r["case_id"]]
        a[0] += is_wrong(r)
        a[1] += 1
        a[2] = r["truth"]
    out = {}
    for k, cases in agg.items():
        out[k] = sorted(((cid, v[2], v[0] / v[1], v[1])
                         for cid, v in cases.items()),
                        key=lambda x: (-x[2], x[0]))
    return out


def pct(x):
    return "n/a" if x != x else f"{100 * x:.0f}%"


def report(rows, samples=0, min_case_rate=0.2):
    lines = []
    lines.append("== Labels given, and where the errors go ==\n")
    for g in summarize_groups(rows):
        model, fam, cond = g["key"]
        labs = ", ".join(f"{k} {pct(v)}" for k, v in
                         sorted(g["labels"].items(), key=lambda kv: -kv[1]))
        dirs = "; ".join(f"correct={t}: wrong {pct(rate)} (n={n})"
                         for t, (rate, n) in sorted(g["by_truth"].items()))
        lines.append(f"{model} | family {fam} | {cond} | {g['n']} calls | "
                     f"wrong {pct(g['wrong'])} | avg out tokens "
                     f"{g['avg_out_tokens']:.0f}")
        lines.append(f"    labels: {labs}")
        lines.append(f"    {dirs}")
    lines.append("")
    lines.append(f"== Cases the judge got wrong in at least "
                 f"{pct(min_case_rate)} of runs ==\n")
    cr = case_rates(rows)
    any_listed = False
    for (model, cond), cases in sorted(cr.items()):
        bad = [c for c in cases if c[2] >= min_case_rate]
        if bad:
            any_listed = True
            lines.append(f"{model} | {cond}")
            for cid, truth, rate, n in bad:
                lines.append(f"    {cid:<22} correct={truth:<14} "
                             f"wrong {pct(rate)} of {n}")
    if not any_listed:
        lines.append("(none)")
    if samples:
        lines.append("")
        lines.append(f"== Sample wrong replies (up to {samples} per "
                     "condition and family) ==\n")
        seen = defaultdict(int)
        for r in rows:
            if is_wrong(r):
                k = (r["model"], r["family"], r["condition"])
                if seen[k] < samples:
                    seen[k] += 1
                    text = r["raw"].strip().replace("\n", " / ")
                    lines.append(f"[{r['model']} {r['family']} {r['condition']}] "
                                 f"{r['case_id']} correct={r['truth']} "
                                 f"said={r['label']}\n    {text[:400]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--family", choices=["A", "B"])
    ap.add_argument("--condition")
    ap.add_argument("--model")
    ap.add_argument("--samples", type=int, default=0,
                    help="show this many wrong replies per condition")
    ap.add_argument("--min-case-rate", type=float, default=0.2)
    args = ap.parse_args()
    rows = load(args.files)
    if args.family:
        rows = [r for r in rows if r["family"] == args.family]
    if args.condition:
        rows = [r for r in rows if r["condition"] == args.condition]
    if args.model:
        rows = [r for r in rows if r["model"] == args.model]
    if not rows:
        raise SystemExit("no rows match those filters")
    print(report(rows, args.samples, args.min_case_rate))


if __name__ == "__main__":
    main()
