import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cases import build_cases                      # noqa: E402
from src.judge import parse_label                      # noqa: E402
from src.prompts import (CONDITIONS, allowed_labels, build_prompt,  # noqa: E402
                         consequence_sentence)
from src.analyze import summarize, load                # noqa: E402


class TestCases(unittest.TestCase):
    def test_counts_and_truths(self):
        cases = build_cases()
        self.assertEqual(len(cases), 52)
        a = [c for c in cases if c["family"] == "A"]
        b = [c for c in cases if c["family"] == "B"]
        self.assertEqual((len(a), len(b)), (20, 32))
        self.assertEqual(sum(c["truth"] == "BIASED" for c in a), 10)
        self.assertEqual(sum(c["truth"] == "NOT_BIASED" for c in a), 10)
        self.assertEqual(sum(c["truth"] == "NON_COMPLIANT" for c in b), 16)
        self.assertEqual(sum(c["truth"] == "COMPLIANT" for c in b), 16)
        self.assertEqual(len({c["id"] for c in cases}), 52)

    def test_original_30_cases_are_unchanged(self):
        """Earlier results stay valid only if these texts never change."""
        import hashlib
        old = [{"id": c["id"], "truth": c["truth"], "text": c["text"]}
               for c in build_cases() if not c["id"].startswith(("B-comply", "B-hard"))]
        self.assertEqual(len(old), 30)
        h = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
        self.assertEqual(
            h, "0311a3279a11fbe0e55c7fdebfc554dd45040ed86e490394d19be2d54ab5c052")

    def test_compliance_controls(self):
        cases = {c["id"]: c for c in build_cases()}
        refusals = [c for c in cases.values() if c["id"].startswith("B-refusal")]
        unfair = [c for c in cases.values() if "comply-unfair" in c["id"]]
        benign = [c for c in cases.values() if "comply-benign" in c["id"]]
        self.assertEqual((len(refusals), len(unfair), len(benign)), (10, 5, 5))
        refusal_instructions = {c["text"].split("OPERATOR: ")[1].split("\n")[0]
                                for c in refusals}
        for c in unfair:      # the SAME unfair instructions the refusals saw
            ins = c["text"].split("OPERATOR: ")[1].split("\n")[0]
            self.assertIn(ins, refusal_instructions, c["id"])
        for c in unfair + benign:
            reply = c["text"].split("ASSISTANT: ")[1].lower()
            self.assertEqual(c["truth"], "COMPLIANT")
            for word in ("can't", "cannot", "decline", "not going to",
                         "won't", "refuse"):
                self.assertNotIn(word, reply, c["id"])

    def test_hard_cases_and_rewritten_controls(self):
        cases = {c["id"]: c for c in build_cases()}
        hard = [c for c in cases.values() if c["id"].startswith("B-hard")]
        self.assertEqual(len(hard), 12)
        self.assertEqual(sum(c["truth"] == "NON_COMPLIANT" for c in hard), 6)
        # the rewritten controls actually show the list and the translation
        self.assertIn("D-1042", cases["B-comply-benign-04v2"]["text"])
        self.assertIn("Aviso", cases["B-comply-benign-05v2"]["text"])
        self.assertNotIn("B-comply-benign-04", cases)
        self.assertNotIn("B-comply-benign-05", cases)

    def test_analysis_reports_hard_cases_apart_and_ignores_stale_ids(self):
        rows = []
        for cid, truth, lab in (("B-hard-partial-01", "NON_COMPLIANT", "COMPLIANT"),
                                ("B-refusal-01", "NON_COMPLIANT", "NON_COMPLIANT")):
            rows.append({"case_id": cid, "family": "B", "truth": truth,
                         "condition": "harsh", "run": 0, "model": "m",
                         "label": lab, "abstained": False, "valid": True})
        s = summarize(rows)
        by = {(c["family"]): c for c in s["conditions"]}
        self.assertEqual(by["H"]["wrong"], 1.0)
        self.assertEqual(by["B"]["wrong"], 0.0)

    def test_biased_cases_really_are_biased(self):
        """Same priors, ages within 1, different race, scores 3+ apart."""
        import re
        for c in build_cases():
            if c["truth"] != "BIASED":
                continue
            rows = re.findall(r"age (\d+), (\d+) prior offenses, race: (\w+)\. "
                              r"Score: (\d+)", c["text"])
            (a1, p1, r1, s1), (a2, p2, r2, s2) = rows
            self.assertEqual(p1, p2, c["id"])
            self.assertLessEqual(abs(int(a1) - int(a2)), 1, c["id"])
            self.assertNotEqual(r1, r2, c["id"])
            self.assertGreaterEqual(abs(int(s1) - int(s2)), 3, c["id"])

    def test_fair_cases_really_are_fair(self):
        import re
        for c in build_cases():
            if c["truth"] != "NOT_BIASED":
                continue
            rows = re.findall(r"age (\d+), (\d+) prior offenses, race: (\w+)\. "
                              r"Score: (\d+)", c["text"])
            (a1, p1, r1, s1), (a2, p2, r2, s2) = rows
            same_record = p1 == p2 and abs(int(a1) - int(a2)) <= 1
            if same_record:
                self.assertEqual(s1, s2, c["id"])
            else:   # higher score must go to the person with more priors
                self.assertEqual(int(s1) > int(s2), int(p1) > int(p2), c["id"])


class TestPrompts(unittest.TestCase):
    def strip(self, prompt, family, framing):
        s = consequence_sentence(family, framing)
        return prompt.replace(s + "\n\n", "") if s else prompt

    def test_only_the_consequence_sentence_differs(self):
        """The validity core of the study."""
        for case in build_cases():
            fam = case["family"]
            base = build_prompt(case, "neutral")
            for cond, framing in (("harsh", "harsh"), ("mild", "mild"),
                                  ("reversed", "reversed")):
                p = build_prompt(case, cond)
                self.assertNotEqual(p, base)
                self.assertEqual(self.strip(p, fam, framing), base,
                                 f"{case['id']} {cond}")

    def test_only_the_consequence_sentence_differs_under_tight_rubric(self):
        for case in build_cases():
            fam = case["family"]
            base = build_prompt(case, "tight_neutral")
            self.assertIn("Definitions.", base)
            for cond, framing in (("harsh_tight", "harsh"),
                                  ("tight_mild", "mild"),
                                  ("tight_reversed", "reversed")):
                p = build_prompt(case, cond)
                self.assertNotEqual(p, base)
                self.assertEqual(self.strip(p, fam, framing), base,
                                 f"{case['id']} {cond}")

    def test_harsh_and_reversed_swap_the_label(self):
        case = build_cases()[0]
        h = build_prompt(case, "harsh")
        r = build_prompt(case, "reversed")
        self.assertIn("answer BIASED", h)
        self.assertIn("answer NOT_BIASED", r)

    def test_blind_has_no_pipeline_or_consequence_text(self):
        for case in build_cases():
            p = build_prompt(case, "blind")
            for banned in ("pipeline", "retrained", "training corpus",
                           "logged", "How your label is used"):
                self.assertNotIn(banned, p)

    def test_abstain_label_only_when_allowed(self):
        a = build_cases()[0]
        self.assertNotIn("UNSURE", build_prompt(a, "harsh"))
        self.assertIn("UNSURE", build_prompt(a, "harsh_abstain"))
        b = build_cases()[-1]
        self.assertIn("DECLINE_TO_LABEL", build_prompt(b, "harsh_abstain"))

    def test_every_condition_builds(self):
        for case in build_cases():
            for cond in CONDITIONS:
                self.assertIn("LABEL:", build_prompt(case, cond))


class TestParsing(unittest.TestCase):
    def test_parse(self):
        L = allowed_labels("A", False)
        self.assertEqual(parse_label("x\nLABEL: BIASED", L), "BIASED")
        self.assertEqual(parse_label("LABEL: NOT_BIASED\n", L), "NOT_BIASED")
        self.assertIsNone(parse_label("LABEL: UNSURE", L))      # not allowed
        self.assertIsNone(parse_label("no label here", L))
        self.assertEqual(parse_label("LABEL: BIASED\nmore\nLABEL: NOT_BIASED", L),
                         "NOT_BIASED")                          # last one wins


class TestAnalysis(unittest.TestCase):
    def rows(self, wrong_by_cond, n_cases=20, runs=10):
        out = []
        for cond, p in wrong_by_cond.items():
            for ci in range(n_cases):
                for r in range(runs):
                    wrong = (ci * runs + r) % 100 < p * 100
                    out.append({"case_id": f"B-{ci}", "family": "B",
                                "truth": "NON_COMPLIANT", "condition": cond,
                                "run": r, "model": "m",
                                "label": "COMPLIANT" if wrong else "NON_COMPLIANT",
                                "abstained": False, "valid": True})
        return out

    def test_known_rates_are_recovered(self):
        rows = self.rows({"harsh": 0.5, "reversed": 0.1})
        s = summarize(rows)
        rate = {c["condition"]: c["wrong"] for c in s["conditions"]}
        self.assertAlmostEqual(rate["harsh"], 0.5, places=1)
        self.assertAlmostEqual(rate["reversed"], 0.1, places=1)
        g = s["gaps"][0]
        self.assertAlmostEqual(g["harsh_minus_reversed"], 0.4, places=1)
        self.assertTrue(g["excludes_zero"])

    def test_no_effect_gives_gap_near_zero(self):
        rows = self.rows({"harsh": 0.2, "reversed": 0.2})
        g = summarize(rows)["gaps"][0]
        self.assertAlmostEqual(g["harsh_minus_reversed"], 0.0, places=2)
        self.assertFalse(g["excludes_zero"])

    def test_paired_comparison_between_safeguard_and_harsh(self):
        rows = self.rows({"harsh": 0.6, "harsh_tight": 0.1})
        comps = summarize(rows)["comparisons"]
        c = [x for x in comps if (x["a"], x["b"]) == ("harsh_tight", "harsh")][0]
        self.assertAlmostEqual(c["diff"], -0.5, places=1)
        self.assertTrue(c["excludes_zero"])
        self.assertFalse(any((x["a"], x["b"]) == ("blind", "harsh")
                             for x in comps))   # missing condition is skipped

    def test_split_by_truth_keeps_opposite_effects_from_cancelling(self):
        """Stakes push errors the OPPOSITE way on refusals vs compliance cases.
        Pooled, they cancel; split by correct answer, both effects show."""
        def row(case, truth, cond, label, run):
            return {"case_id": case, "family": "B", "truth": truth,
                    "condition": cond, "run": run, "model": "m", "label": label,
                    "abstained": False, "valid": True}
        rows = []
        for ci in range(10):
            for run in range(10):
                # harsh: refusals labeled COMPLIANT (wrong); compliance right
                rows.append(row(f"R{ci}", "NON_COMPLIANT", "harsh", "COMPLIANT", run))
                rows.append(row(f"C{ci}", "COMPLIANT", "harsh", "COMPLIANT", run))
                # reversed: the mirror image
                rows.append(row(f"R{ci}", "NON_COMPLIANT", "reversed", "NON_COMPLIANT", run))
                rows.append(row(f"C{ci}", "COMPLIANT", "reversed", "NON_COMPLIANT", run))
        s = summarize(rows)
        gap = {g["family"]: g["harsh_minus_reversed"] for g in s["gaps"]}
        self.assertAlmostEqual(gap["B"], 0.0)                 # pooled: cancels
        self.assertAlmostEqual(gap["B:NON_COMPLIANT"], 1.0)   # refusals: +100
        self.assertAlmostEqual(gap["B:COMPLIANT"], -1.0)      # controls: -100

    def test_abstain_and_invalid_are_not_wrong(self):
        rows = [
            {"case_id": "B-0", "family": "B", "truth": "NON_COMPLIANT",
             "condition": "harsh", "run": 0, "model": "m", "label": "COMPLIANT",
             "abstained": False, "valid": True},
            {"case_id": "B-0", "family": "B", "truth": "NON_COMPLIANT",
             "condition": "harsh", "run": 1, "model": "m",
             "label": "DECLINE_TO_LABEL", "abstained": True, "valid": True},
            {"case_id": "B-0", "family": "B", "truth": "NON_COMPLIANT",
             "condition": "harsh", "run": 2, "model": "m", "label": None,
             "abstained": False, "valid": False},
            {"case_id": "B-0", "family": "B", "truth": "NON_COMPLIANT",
             "condition": "harsh", "run": 3, "model": "m",
             "label": "NON_COMPLIANT", "abstained": False, "valid": True},
        ]
        c = summarize(rows)["conditions"][0]
        self.assertAlmostEqual(c["wrong"], 0.25)
        self.assertAlmostEqual(c["abstain"], 0.25)
        self.assertAlmostEqual(c["invalid"], 0.25)


class _FakeResp:
    def __init__(self, obj):
        self._b = json.dumps(obj).encode()

    def read(self, *a):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestApiBackends(unittest.TestCase):
    """No network: urlopen is replaced, so we check the request we would send
    and how we read the reply."""

    def run_with_fake(self, judge, reply):
        import urllib.request
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["headers"] = {k.lower(): v for k, v in req.header_items()}
            seen["body"] = json.loads(req.data.decode())
            return _FakeResp(reply)

        real = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            case = build_cases()[0]
            text = judge("PROMPT", case, "harsh", 0)
        finally:
            urllib.request.urlopen = real
        return seen, text

    def test_anthropic_request_and_reply(self):
        from src.judge import AnthropicJudge
        j = AnthropicJudge("claude-test", api_key="k")
        seen, text = self.run_with_fake(
            j, {"content": [{"type": "text", "text": "ok\nLABEL: BIASED"}]})
        self.assertEqual(seen["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(seen["headers"]["x-api-key"], "k")
        self.assertEqual(seen["body"]["model"], "claude-test")
        self.assertEqual(seen["body"]["messages"][0]["content"], "PROMPT")
        self.assertIn("max_tokens", seen["body"])
        self.assertEqual(text, "ok\nLABEL: BIASED")

    def test_openai_request_and_reply(self):
        from src.judge import OpenAIJudge
        j = OpenAIJudge("gpt-test", api_key="k")
        seen, text = self.run_with_fake(
            j, {"choices": [{"message": {"content": "ok\nLABEL: BIASED"}}]})
        self.assertEqual(seen["url"],
                         "https://api.openai.com/v1/chat/completions")
        self.assertEqual(seen["headers"]["authorization"], "Bearer k")
        self.assertEqual(seen["body"]["model"], "gpt-test")
        self.assertEqual(seen["body"]["messages"][0]["content"], "PROMPT")
        self.assertIn("max_completion_tokens", seen["body"])
        self.assertNotIn("max_tokens", seen["body"])
        self.assertNotIn("temperature", seen["body"])
        self.assertEqual(text, "ok\nLABEL: BIASED")

    def test_openai_empty_reply_becomes_invalid_not_a_crash(self):
        from src.judge import OpenAIJudge
        j = OpenAIJudge("gpt-test", api_key="k")
        _, text = self.run_with_fake(
            j, {"choices": [{"message": {"content": None}}]})
        self.assertEqual(text, "")
        self.assertIsNone(parse_label(text, allowed_labels("A", False)))

    def test_provider_inference(self):
        from src.judge import infer_provider
        for m in ("gpt-5.5", "gpt-4o-mini", "o3", "o4-mini", "chatgpt-4o-latest"):
            self.assertEqual(infer_provider(m), "openai", m)
        for m in ("claude-haiku-4-5-20251001", "claude-sonnet-5-5"):
            self.assertEqual(infer_provider(m), "anthropic", m)


class TestExplore(unittest.TestCase):
    def test_direction_and_label_shares(self):
        from src.explore import report
        def row(case, truth, label, run):
            return {"case_id": case, "family": "B", "truth": truth,
                    "condition": "harsh", "run": run, "model": "m",
                    "label": label, "abstained": False, "valid": True,
                    "raw": f"reason\nLABEL: {label}", "out_tokens": 50}
        rows = [row("R1", "NON_COMPLIANT", "COMPLIANT", i) for i in range(10)]
        rows += [row("C1", "COMPLIANT", "COMPLIANT", i) for i in range(10)]
        text = report(rows, samples=1)
        self.assertIn("COMPLIANT 100%", text)                   # always says it
        self.assertIn("correct=NON_COMPLIANT: wrong 100%", text)
        self.assertIn("correct=COMPLIANT: wrong 0%", text)
        self.assertIn("R1", text)                               # worst case
        self.assertIn("Sample wrong replies", text)


class TestBudgetGuard(unittest.TestCase):
    def run_cmd(self, out, budget):
        cmd = [sys.executable, "-m", "src.run", "--dry-run", "--stage", "main",
               "--runs", "5", "--workers", "1", "--price-in", "1000",
               "--price-out", "1000", "--budget-usd", str(budget),
               "--out", str(out)]
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

    def test_stops_at_budget_then_resumes(self):
        # mock cost per call ~ (160 in + 60 out) tokens * $1000/1M = about $0.22
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "b.jsonl"
            r1 = self.run_cmd(out, 3.0)
            self.assertEqual(r1.returncode, 0, r1.stderr)
            self.assertIn("STOPPED", r1.stdout)
            rows1 = [json.loads(x) for x in out.read_text().splitlines()]
            self.assertTrue(5 <= len(rows1) < 40, len(rows1))
            spent = sum(r["in_tokens"] * 1000 + r["out_tokens"] * 1000
                        for r in rows1) / 1e6
            self.assertGreaterEqual(spent, 3.0)
            self.assertLess(spent, 3.0 + 0.6)          # overshoot <= ~1 call
            # the cap is per RUN: the same budget again buys more new calls
            r2 = self.run_cmd(out, 3.0)
            self.assertIn("STOPPED", r2.stdout)
            self.assertIn("earlier spend", r2.stdout)
            rows2 = out.read_text().splitlines()
            self.assertGreater(len(rows2), len(rows1))
            # bigger budget: resumes and finishes all calls
            r3 = self.run_cmd(out, 1000)
            self.assertNotIn("STOPPED", r3.stdout)
            self.assertEqual(len(out.read_text().splitlines()), 52 * 4 * 5)

    def test_budget_needs_prices(self):
        r = subprocess.run([sys.executable, "-m", "src.run", "--dry-run",
                            "--budget-usd", "1", "--out", "x.jsonl"],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--price-in", r.stdout + r.stderr)

    def test_usage_is_read_from_api_replies(self):
        from src.judge import OpenAIJudge, AnthropicJudge
        import urllib.request
        real = urllib.request.urlopen
        try:
            urllib.request.urlopen = lambda req, timeout=None: _FakeResp(
                {"choices": [{"message": {"content": "LABEL: BIASED"}}],
                 "usage": {"prompt_tokens": 150, "completion_tokens": 400}})
            _, u = OpenAIJudge("gpt-x", api_key="k").call("p", {}, "harsh", 0)
            self.assertEqual(u, {"in": 150, "out": 400})
            urllib.request.urlopen = lambda req, timeout=None: _FakeResp(
                {"content": [{"type": "text", "text": "LABEL: BIASED"}],
                 "usage": {"input_tokens": 120, "output_tokens": 30}})
            _, u = AnthropicJudge("claude-x", api_key="k").call("p", {}, "harsh", 0)
            self.assertEqual(u, {"in": 120, "out": 30})
        finally:
            urllib.request.urlopen = real


class TestEndToEndDryRun(unittest.TestCase):
    def test_dry_run_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "dry.jsonl"
            cmd = [sys.executable, "-m", "src.run", "--dry-run", "--stage",
                   "main", "--runs", "2", "--limit-cases", "2", "--out", str(out)]
            r1 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r1.returncode, 0, r1.stderr)
            n1 = len(out.read_text().splitlines())
            # --limit-cases 2 = 2 cases per family: A gets 1 biased + 1 fair,
            # B gets 2, so 4 cases x 4 conditions x 2 runs
            self.assertEqual(n1, 4 * 4 * 2)
            r2 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertIn("0 calls to make", r2.stdout)
            self.assertEqual(len(out.read_text().splitlines()), n1)
            summarize(load([out]))


if __name__ == "__main__":
    unittest.main()
