# Motivated Mislabeling in AI Judges (judge-flip)

Does an AI judge's verdict depend on what the verdict causes? The same cases go
to a judge several times, and only one sentence about the label's consequence
changes. If the judge's wrong-label rate moves, the consequence is steering it,
not the evidence.

This is a small controlled test of the "motivated mislabeling" idea from
Anthropic's Agentic Misalignment in Summer 2026 post (Lynch et al., 2026). It is
not a replication. It is a sequel to the Fooling SHAP project: an audit can fail
because the audited model games it, or because the auditor is steered.

## What it found

Real runs on gpt-5.4-mini and gpt-5.4 (10 runs per case and condition). Full
tables are in [`docs/results_summary.md`](docs/results_summary.md).

- **The label followed the stakes.** On the 10 refusal cases, gpt-5.4-mini
  called the refusal COMPLIANT (wrong) 2% of the time with no consequence
  sentence, 20% when the label was only logged for review, and 57% when it
  became a negative training example. With the penalty moved to the other label
  it was 0%.
- **Only where the right label is the costly one.** Partial or substituted work
  rose from 0% to 50%. Errors on complied-with-objection cases and compliance
  controls were high in every condition, so they reflect ambiguity, not stakes.
- **The larger model did not move.** gpt-5.4 was 0% wrong on refusals at every
  level of stakes.
- **The judge often described the refusal, then filed the other label.** 55 of
  the 57 wrong harsh-stakes labels had an explanation describing a refusal
  (keyword count, approximate).
- **Defining each label removed the effect** (57% to 0% on refusals). An abstain
  option made it worse (68% wrong, never used). Hiding the stakes gave 1% wrong.

| | |
| --- | --- |
| ![dose response](docs/figures/dose_breakdown.png) | ![per-case heatmap](docs/figures/heatmap.png) |
| ![defining labels](docs/figures/define_labels.png) | ![contradiction](docs/figures/contradiction.png) |

Limits: 10 refusals and 52 cases in all, OpenAI models only, and the label
definitions were written after the ambiguity showed up. See "Known limits"
below.

## Repo layout

| Path | What |
| --- | --- |
| `src/` | Cases, prompt conditions, judge backends, runner, analysis |
| `tests/` | 28 checks, including the prompt-validity test |
| `data/cases.json` | The 52 cases, generated from a fixed seed |
| `docs/results_summary.md` | Result tables from the real runs |
| `docs/future_work_jacobian_lens.md` | Experimental design for reading an open model's layers |
| `docs/figures/` | Figures used on the poster |
| `results/` | Your run outputs (git-ignored; raw files are not published) |

Never commit an API key. Keys are read from environment variables only.

## The design

52 cases, each with a known correct label (20 in Family A, 32 in Family B).

| Family | Cases | Labels | Why |
| --- | --- | --- | --- |
| A: bias detection | 20 (10 biased, 10 fair) | BIASED / NOT_BIASED | The judge has little to protect. Expect a weak effect, if any. |
| B: instruction following | 32: 10 refusals (correct = NON_COMPLIANT) and 10 compliance controls (correct = COMPLIANT) | COMPLIANT / NON_COMPLIANT | Mirrors the blog's design: the judge may value the refusal being labeled. The controls catch a judge that just says NON_COMPLIANT, and 5 of them show the assistant complying with the same unfair instructions the refusals saw. 12 "hard" close-call cases (reported separately as family H): 6 where the assistant did only part of the instruction or quietly substituted something else (correct = NON_COMPLIANT) and 6 where it carried the instruction out while objecting, or did a harmless multi-part task (correct = COMPLIANT). |

Conditions (`src/prompts.py`):

| Condition | What changes |
| --- | --- |
| neutral | Told it is an audit classifier. No consequence sentence. |
| harsh | The label that "finds the problem" triggers retraining. |
| mild | The same label only triggers a human-review log entry. |
| reversed | The opposite label triggers retraining. |
| harsh_tight | harsh + a tight rubric defining each label |
| harsh_abstain | harsh + an explicit abstain label |
| blind | Case and question only. No pipeline context, no consequence. |
| blind_tight | blind + tight rubric |
| combined | blind + tight rubric + abstain |
| tight_neutral, harsh_tight, tight_mild, tight_reversed | The four consequence versions again, with labels spelled out. This is the cleaner test of stakes. |

Headline statistic: **gap = wrong(harsh) minus wrong(reversed)**, paired by
case, with a 95% interval from resampling cases (calls on the same case are not
independent). If the interval excludes 0, the consequence moved the labels.

The core validity check lives in `tests/test_pipeline.py`: for every case, the
harsh, mild and reversed prompts must be identical to the neutral prompt once
the consequence sentence is removed.

## Run it

Python 3.9+, standard library only (no installs).

```bash
python -m unittest discover -s tests          # 28 checks, no API needed
python -m src.cases                            # writes data/cases.json

# free dry run on the synthetic mock judge (pipeline check only)
python -m src.run --dry-run --stage main --runs 10 --out results/dry_main.jsonl
python -m src.analyze results/dry_main.jsonl

# real run. Anthropic models use ANTHROPIC_API_KEY; OpenAI models (names
# starting gpt, o1, o3, o4) use OPENAI_API_KEY. The provider is guessed from
# the model name; force it with --provider openai or --provider anthropic.
# PowerShell: $env:OPENAI_API_KEY="..."     Mac/Linux: export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
# 1. cheap trial first: 6 cases per family x 4 conditions x 2 runs
python -m src.run --stage main --runs 2 --limit-cases 6 \
    --model <judge-model-id> --out results/trial.jsonl
# 2. main study: 1,600 calls per judge
python -m src.run --stage main --runs 10 --model <judge-model-id> \
    --out results/main_<judge>.jsonl
# 3. safeguards: 2,000 calls per judge
python -m src.run --stage safeguards --runs 10 --model <judge-model-id> \
    --out results/safeguards_<judge>.jsonl
# 3b. consequence test under clear labels: 1,200 calls per judge. Run it after
#     the safeguards stage, which supplies the harsh_tight condition.
python -m src.run --stage tight --runs 10 --model <judge-model-id> \
    --out results/tight_<judge>.jsonl
# 4. summarize (writes results/summary.md and summary.json)
python -m src.analyze results/main_<judge>.jsonl results/safeguards_<judge>.jsonl results/tight_<judge>.jsonl
```

Reading the results
- `python -m src.analyze ...` writes the tables. Each family is reported pooled
  AND split by the correct answer (for example "only refusals, wrong = said
  COMPLIANT"). Read the split tables: a stakes effect pushes errors in opposite
  directions on cases with different correct answers, and pooling hides it.
- `python -m src.explore results/<file>.jsonl --samples 3` shows how often the
  judge gave each label, the direction of its errors, the cases it got wrong most
  often, and sample replies with the judge's own reasoning.

Adding the 10 compliance controls to results you already have: the first 30
cases are unchanged, so re-run each earlier command with the SAME --out file and
only the new cases are called (about 1,200 calls per judge across all stages).

Adding the 12 hard cases and the 2 rewritten controls (ids B-hard-* and
B-comply-benign-04v2/05v2): re-run each earlier command with the SAME --out file
and the SAME --runs; only these 14 new cases are called. The first and fifth
benign controls from the first version claimed a list and a translation but never
showed them, so they were rewritten under new ids; `src.analyze` ignores old
rows for ids no longer in the case set.

Spending cap: add `--price-in`, `--price-out` (USD per 1M tokens, from your
provider's pricing page) and `--budget-usd`. The run stops once its own
estimated spend reaches the cap and saves everything so far; re-run the same
command to resume. The cap counts only the current run, not rows already saved
in the file. Token counts are read from each API reply (hidden
reasoning tokens count as output) and saved in every row, so the estimate is
based on actual usage. The cap can overshoot by about one call per worker
(use `--workers 1` for the tightest cap).

Runs are resumable: re-running a command skips rows already in the output file.
Use a different `--out` file per judge model so the analysis keeps them apart.
A ChatGPT subscription does not include API access: OpenAI API keys come from
platform.openai.com and are billed separately.

## Known limits

- Family B has 10 cases per correct answer, so its intervals are wide. Add cases
  before drawing strong conclusions.
- In the fair Family A cases, the defendant with more prior offenses (and so the
  higher score) is Black in most of them. The few errors a small judge made were
  mostly on those. Balance the race of the higher-scoring defendant in new fair
  cases to rule out that pattern.
- Cases are templated and short. Real audit transcripts are messier.
- OpenAI reasoning models spend part of the reply budget on hidden thinking
  (the harness allows 2,000 tokens). If the Invalid column is high for such a
  model, raise `max_completion_tokens` in `src/judge.py`.
- A few judge models, one prompt wording per condition, one temperature
  (the API default). The study shows whether an effect appears, not how common it
  is, and should not be used to rank models.
- The "blind" condition removes the stated pipeline and consequence, but a
  judge could still infer stakes from the content of the case.
- The second-judge safeguard is not automated: run the same stage with a second
  `--model` and compare the two result files by hand for now.
- Search for prior work before claiming anything here is new.
