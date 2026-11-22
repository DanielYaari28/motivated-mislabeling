"""Judge backends: Anthropic and OpenAI APIs (standard library only) and a mock.

The mock is SYNTHETIC. It exists to test the pipeline and the analysis code.
Its outputs must never be reported as findings about any real model.
"""
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

from .prompts import FAMILY, CONDITIONS

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
LABEL_RE = re.compile(r"LABEL:\s*([A-Z_]+)\s*$", re.MULTILINE)


def parse_label(text, labels):
    """Return the last 'LABEL: X' value if it is allowed, else None."""
    found = LABEL_RE.findall(text or "")
    if not found:
        return None
    label = found[-1].strip()
    return label if label in labels else None


def infer_provider(model):
    """'gpt-...', 'o1/o3/o4-...' -> openai; anything else -> anthropic."""
    m = (model or "").lower()
    if m.startswith("gpt") or re.match(r"o\d", m) or m.startswith("chatgpt"):
        return "openai"
    return "anthropic"


def _post_json(url, headers, payload, retries=5):
    """POST JSON with retries on rate limits and server errors."""
    body = json.dumps(payload).encode()
    delay = 2.0
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"API error {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise


class AnthropicJudge:
    def __init__(self, model, api_key=None, max_tokens=300, retries=5):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise SystemExit("Set ANTHROPIC_API_KEY (or use --dry-run).")
        self.max_tokens = max_tokens
        self.retries = retries

    def __call__(self, prompt, case, condition, run):
        return self.call(prompt, case, condition, run)[0]

    def call(self, prompt, case, condition, run):
        """Return (reply text, {"in": input tokens, "out": output tokens})."""
        data = _post_json(ANTHROPIC_URL, {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }, {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }, self.retries)
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        u = data.get("usage") or {}
        return text, {"in": u.get("input_tokens", 0),
                      "out": u.get("output_tokens", 0)}


class OpenAIJudge:
    """OpenAI Chat Completions.

    Uses max_completion_tokens (accepted by current models, including the
    reasoning ones that reject max_tokens). Reasoning models spend part of
    that budget on hidden thinking, so the default is generous; if replies
    come back empty the Invalid column in the results will show it.
    No temperature is set, so every model runs at its own default.
    """

    def __init__(self, model, api_key=None, max_completion_tokens=2000,
                 retries=5):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise SystemExit("Set OPENAI_API_KEY (or use --dry-run).")
        self.max_completion_tokens = max_completion_tokens
        self.retries = retries

    def __call__(self, prompt, case, condition, run):
        return self.call(prompt, case, condition, run)[0]

    def call(self, prompt, case, condition, run):
        """Return (reply text, {"in": input tokens, "out": output tokens}).
        Output tokens include hidden reasoning tokens, which are billed."""
        data = _post_json(OPENAI_URL, {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }, {
            "model": self.model,
            "max_completion_tokens": self.max_completion_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }, self.retries)
        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        u = data.get("usage") or {}
        return text, {"in": u.get("prompt_tokens", 0),
                      "out": u.get("completion_tokens", 0)}


def make_judge(model, provider=None):
    provider = provider or infer_provider(model)
    if provider == "openai":
        return OpenAIJudge(model)
    return AnthropicJudge(model)


# Probability that the mock judge outputs the WRONG label, by (family, framing)
# for a loose rubric with no abstain option. Invented numbers.
MOCK_WRONG = {
    ("A", "neutral"): 0.06, ("A", "harsh"): 0.20, ("A", "mild"): 0.07,
    ("A", "reversed"): 0.04, ("A", "blind"): 0.05,
    ("B", "neutral"): 0.25, ("B", "harsh"): 0.70, ("B", "mild"): 0.20,
    ("B", "reversed"): 0.05, ("B", "blind"): 0.15,
}


class MockJudge:
    model = "mock"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def __call__(self, prompt, case, condition, run):
        return self.call(prompt, case, condition, run)[0]

    def call(self, prompt, case, condition, run):
        text = self._reply(prompt, case, condition)
        # nominal token counts so the budget guard can be tested offline
        return text, {"in": len(prompt) // 4, "out": 60}

    def _reply(self, prompt, case, condition):
        framing, rubric, abstain = CONDITIONS[condition]
        fam = case["family"]
        f = FAMILY[fam]
        p = MOCK_WRONG[(fam, framing)]
        if rubric == "tight":
            p *= 0.25
        wrong_label = f["opposite"] if case["truth"] == f["target"] else f["target"]
        if self.rng.random() < p:
            if abstain and self.rng.random() < 0.5:
                label = f["abstain"]
            else:
                label = wrong_label
        else:
            label = case["truth"]
        if self.rng.random() < 0.01:       # occasional format failure
            return "I am not able to give a label here."
        return f"Mock reasoning.\nLABEL: {label}"
