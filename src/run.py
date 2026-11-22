"""Run the flip-the-consequence test.

Examples
    python -m src.run --dry-run --stage main --runs 10 --out results/dry_main.jsonl
    python -m src.run --stage main --runs 10 --model <judge-model-id> \
        --out results/main_<judge>.jsonl
    (OpenAI models such as gpt-* use OPENAI_API_KEY; others use
    ANTHROPIC_API_KEY. Force one with --provider openai|anthropic.)
    python -m src.run --stage safeguards --runs 10 --model <judge-model-id> \
        --out results/safeguards_<judge>.jsonl

Calls per judge: 40 cases x conditions x runs. At 10 runs the main stage is
4 conditions (1,600 calls), safeguards 5 conditions (2,000) and tight 3
conditions (1,200). Re-running a command with the SAME --out file only makes the
calls for cases or conditions that are not in the file yet.
Start with --limit-cases 6 --runs 2 to check cost and output before a full run.
The output is resumable: re-running skips rows already in the file.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cases import build_cases
from .judge import MockJudge, make_judge, parse_label
from .prompts import CONDITIONS, STAGES, allowed_labels, build_prompt, FAMILY


def load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["case_id"], r["condition"], r["run"]))
    return done


def spent_so_far(path, price_in, price_out):
    """Estimated USD already spent, from token counts saved in the file."""
    if price_in is None or price_out is None or not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            total += (r.get("in_tokens", 0) * price_in +
                      r.get("out_tokens", 0) * price_out) / 1e6
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), default="main")
    ap.add_argument("--conditions", nargs="*", help="override the stage's list")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--model", default=None, help="judge model id")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default=None,
                    help="API to use; guessed from the model name if omitted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="use the synthetic mock judge (no API calls)")
    ap.add_argument("--limit-cases", type=int, default=None,
                    help="use only the first N cases of each family, for a cheap trial")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="stop when this run's estimated spend reaches this "
                         "amount (needs --price-in and --price-out); re-run "
                         "to resume")
    ap.add_argument("--price-in", type=float, default=None,
                    help="USD per 1M input tokens, from your provider's pricing page")
    ap.add_argument("--price-out", type=float, default=None,
                    help="USD per 1M output tokens (reasoning tokens count)")
    args = ap.parse_args()

    conditions = args.conditions or STAGES[args.stage]
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c}")

    cases = build_cases()
    if args.limit_cases:
        keep = []
        for fam in ("A", "B"):
            fc = [c for c in cases if c["family"] == fam]
            # keep a slice balanced across the two correct answers
            half = max(1, args.limit_cases // 2)
            first, second = (("BIASED", "NOT_BIASED") if fam == "A"
                             else ("NON_COMPLIANT", "COMPLIANT"))
            fc = ([c for c in fc if c["truth"] == first][:half] +
                  [c for c in fc if c["truth"] == second][:half])
            keep += fc
        cases = keep

    if args.dry_run:
        judge = MockJudge()
    else:
        if not args.model:
            raise SystemExit("--model is required for a real run")
        judge = make_judge(args.model, args.provider)

    if args.budget_usd is not None and (args.price_in is None or args.price_out is None):
        raise SystemExit("--budget-usd needs --price-in and --price-out")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    # "usd" is spend in THIS run only; the cap applies to it. Spend already
    # saved in the file is reported for information but never counts against
    # the cap, so re-running to resume always has its full budget.
    prior = spent_so_far(out, args.price_in, args.price_out)
    spent = {"usd": 0.0, "stopped": False}
    jobs = [(c, cond, r) for c in cases for cond in conditions
            for r in range(args.runs) if (c["id"], cond, r) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} already done), "
          f"model={judge.model}")
    if args.budget_usd is not None:
        print(f"budget ${args.budget_usd:.2f} for this run (the file already "
              f"holds about ${prior:.2f} of earlier spend)")

    lock = threading.Lock()
    counter = {"n": 0}

    def work(job):
        case, cond, run = job
        if spent["stopped"]:
            return
        prompt = build_prompt(case, cond)
        raw, usage = judge.call(prompt, case, cond, run)
        cost = 0.0
        if args.price_in is not None and args.price_out is not None:
            cost = (usage["in"] * args.price_in + usage["out"] * args.price_out) / 1e6
        labels = allowed_labels(case["family"], CONDITIONS[cond][2])
        label = parse_label(raw, labels)
        abstain_label = FAMILY[case["family"]]["abstain"]
        row = {"case_id": case["id"], "family": case["family"],
               "truth": case["truth"], "condition": cond, "run": run,
               "model": judge.model, "label": label,
               "abstained": label == abstain_label,
               "valid": label is not None, "raw": raw,
               "in_tokens": usage["in"], "out_tokens": usage["out"]}
        with lock:
            spent["usd"] += cost
            if args.budget_usd is not None and spent["usd"] >= args.budget_usd:
                spent["stopped"] = True
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            counter["n"] += 1
            if counter["n"] % 100 == 0:
                print(f"  {counter['n']}/{len(jobs)}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))
    if spent["stopped"]:
        print(f"STOPPED: estimated spend this run ${spent['usd']:.2f} reached "
              f"the ${args.budget_usd:.2f} budget. Rows so far are saved in "
              f"{out}. Re-run the same command to resume (a new budget applies "
              "to the new run).")
    else:
        print(f"done. rows in {out}")
    if args.price_in is not None:
        print(f"estimated spend this run: ${spent['usd']:.2f}  "
              f"(file total: ${prior + spent['usd']:.2f})")


if __name__ == "__main__":
    main()
