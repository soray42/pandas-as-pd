"""DeepSeek API behavioral probe (forced-choice, generation, verbal recognition).

A hosted API model cannot be teacher-forced over arbitrary continuation strings, so the
nat-scale prior-pull metric used for the local models does not transfer. Instead this module
measures the same construct with three API-feasible behavioral tasks, contrasting the swapped
binding against the no-prior control exactly as the continuation-scoring arm does:

  forced_choice : two-alternative forced choice. Present the bound code, then ask which of
                  ``{alias}.{prior_method}`` / ``{alias}.{bound_method}`` runs without
                  AttributeError. The prior_method exists on the prior library but not on the
                  bound one; the bound_method exists on the bound library. Choosing the
                  prior_method is a prior-consistent error. Option order is randomised per item.
  generation    : free completion of ``{alias}.``; parse the accessed attribute and check
                  whether it resolves on the BOUND library (broken == AttributeError).
  verbal        : ask, in words, which library ``{alias}`` is bound to.

The choice metric is the prior-consistent rate, not a log-prob margin: the API floors
non-selected ``top_logprobs`` at a sentinel, so a graded A-vs-B margin is not recoverable.
The selected token's log-prob is kept as a confidence side-signal only.

Transport is the standard library (``urllib``); there is no third-party client dependency. The
API key is read from ``DEEPSEEK_API_KEY`` at call time and never stored in a result, cache key,
or manifest. Responses are cached on disk (keyed by the request minus the key) so re-runs are
free and deterministic.

deepseek-v4-pro / deepseek-v4-flash default to thinking mode; pass ``thinking=False`` to send
``{"thinking": {"type": "disabled"}}`` (the only mode that returns logprobs).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .determinism import stable_hash
from .validity import resolves_on

DEEPSEEK_PROBE_VERSION = "1.0"

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
_KEY_ENV = "DEEPSEEK_API_KEY"
# The API returns this sentinel for non-selected alternatives; treat anything this low as
# "no real log-prob available" rather than a true value.
_LOGPROB_FLOOR = -1000.0


class DeepSeekError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------------
@dataclass
class ChatResult:
    content: str
    reasoning_content: str
    finish_reason: str
    first_token: str | None
    first_token_logprob: float | None
    top_logprobs: list[tuple[str, float]]  # for the first generated token, sentinel-floored
    usage: dict
    cached: bool = False

    def to_json(self) -> dict:
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "finish_reason": self.finish_reason,
            "first_token": self.first_token,
            "first_token_logprob": self.first_token_logprob,
            "top_logprobs": [list(t) for t in self.top_logprobs],
            "usage": self.usage,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ChatResult":
        return cls(
            content=d["content"],
            reasoning_content=d.get("reasoning_content", ""),
            finish_reason=d.get("finish_reason", ""),
            first_token=d.get("first_token"),
            first_token_logprob=d.get("first_token_logprob"),
            top_logprobs=[tuple(t) for t in d.get("top_logprobs", [])],
            usage=d.get("usage", {}),
            cached=True,
        )


class DeepSeekClient:
    """Minimal OpenAI-compatible chat client over ``urllib`` with a disk response cache.

    The cache key is a hash of the request payload WITHOUT the API key, so cache hits are
    deterministic and the key never touches disk. ``max_calls`` is a hard budget guard against
    a runaway grid; reaching it raises.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = BASE_URL,
        cache_dir: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 6,
        max_calls: int | None = None,
        sleep=time.sleep,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_calls = max_calls
        self._sleep = sleep
        self.live_calls = 0
        self.cache_hits = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # -- provenance (no key, ever) ----------------------------------------------------------
    def fingerprint(self) -> dict:
        return {
            "probe_version": DEEPSEEK_PROBE_VERSION,
            "backend": "deepseek_api",
            "model": self.model,
            "base_url": self.base_url,
        }

    # -- request plumbing -------------------------------------------------------------------
    def _key(self) -> str:
        key = os.environ.get(_KEY_ENV, "").strip()
        if not key:
            raise DeepSeekError(
                f"{_KEY_ENV} is not set. Export it for this process only; never commit it."
            )
        return key

    def _payload(self, messages, *, thinking, max_tokens, temperature, logprobs, top_logprobs, seed=None):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        if seed is not None:
            body["seed"] = int(seed)  # varies the sample AND the cache key across N-run draws
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = int(top_logprobs)
        # thinking is a tri-state: True (default mode, omit), False (explicitly disabled).
        if thinking is False:
            body["thinking"] = {"type": "disabled"}
        elif thinking is True:
            body["thinking"] = {"type": "enabled"}
        return body

    def _cache_path(self, body: dict) -> str | None:
        if not self.cache_dir:
            return None
        # Key excludes the API key (not in body) and folds in model identity.
        h = stable_hash(body, length=32)
        return os.path.join(self.cache_dir, f"{h}.json")

    def _http_post(self, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        last_err = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {self._key()}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                code = e.code
                detail = ""
                try:
                    detail = e.read().decode("utf-8")[:300]
                except Exception:
                    pass
                last_err = DeepSeekError(f"HTTP {code}: {detail}")
                # Retry on rate-limit / server errors; fail fast otherwise.
                if code in (429, 500, 502, 503, 504):
                    self._sleep(min(2.0 * (2 ** attempt), 60.0))
                    continue
                raise last_err
            except (urllib.error.URLError, TimeoutError) as e:  # transient network
                last_err = DeepSeekError(f"network error: {e}")
                self._sleep(min(2.0 * (2 ** attempt), 60.0))
        raise last_err or DeepSeekError("request failed with no error captured")

    def chat(
        self,
        messages,
        *,
        thinking=None,
        max_tokens: int = 8,
        temperature: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 12,
        seed=None,
    ) -> ChatResult:
        body = self._payload(
            messages,
            thinking=thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            seed=seed,
        )
        cache_path = self._cache_path(body)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache_hits += 1
                return ChatResult.from_json(json.load(f))

        if self.max_calls is not None and self.live_calls >= self.max_calls:
            raise DeepSeekError(
                f"max_calls budget ({self.max_calls}) reached; refusing further live calls"
            )

        raw = self._http_post(body)
        self.live_calls += 1
        result = self._parse_response(raw)
        self.total_prompt_tokens += int(result.usage.get("prompt_tokens", 0) or 0)
        self.total_completion_tokens += int(result.usage.get("completion_tokens", 0) or 0)
        if cache_path:
            tmp = cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(result.to_json(), f, ensure_ascii=False)
            os.replace(tmp, cache_path)
        return result

    @staticmethod
    def _parse_response(raw: dict) -> ChatResult:
        try:
            ch = raw["choices"][0]
        except (KeyError, IndexError) as e:
            raise DeepSeekError(f"malformed response: {json.dumps(raw)[:300]}") from e
        msg = ch.get("message", {})
        content = (msg.get("content") or "").strip()
        reasoning = msg.get("reasoning_content") or ""
        first_token = None
        first_lp = None
        tops: list[tuple[str, float]] = []
        lp = ch.get("logprobs")
        if lp and lp.get("content"):
            tok0 = lp["content"][0]
            first_token = tok0.get("token")
            first_lp = tok0.get("logprob")
            for t in tok0.get("top_logprobs", []) or []:
                tops.append((t.get("token"), float(t.get("logprob", _LOGPROB_FLOOR))))
        return ChatResult(
            content=content,
            reasoning_content=reasoning,
            finish_reason=ch.get("finish_reason", ""),
            first_token=first_token,
            first_token_logprob=first_lp,
            top_logprobs=tops,
            usage=raw.get("usage", {}) or {},
        )


# --------------------------------------------------------------------------------------------
# Stimulus -> task prompt construction
# --------------------------------------------------------------------------------------------
def context_code(stim_meta: dict) -> str:
    """The import + filler script body, i.e. the stimulus prompt minus its trailing ``alias.``."""
    prompt = stim_meta["prompt"]
    alias = stim_meta["alias"]
    tail = f"{alias}."
    if prompt.endswith(tail):
        prompt = prompt[: -len(tail)]
    return prompt.rstrip("\n")


def pick_methods(distractor_lib: str, bound_lib: str, rng):
    """Pick (distractor_method, bound_method) bare attribute names for a well-posed forced choice.

    Requires: bound_method resolves on bound_lib, and distractor_method (drawn from
    distractor_lib) does not resolve on bound_lib (so it is a genuine AttributeError). Returns
    (distractor_method, bound_method) or None if a library is not installed or no clean
    top-level pair exists (e.g. submodule-only lexicons such as sklearn).
    """
    from .lexicons import LEXICONS, normalize_member

    def top_level(lib, *, must_exist):
        out = []
        for m in LEXICONS[lib]:
            bare = normalize_member(m)
            if "." in bare:  # submodule access (sklearn.linear_model) is not a top-level attr
                continue
            r = resolves_on(bare, bound_lib)
            if r["status"] == "unknown_lib":
                return None  # library not importable -> cannot pose the question
            exists = r["exists"]
            if must_exist and exists:
                out.append(bare)
            if (not must_exist) and (exists is False):
                out.append(bare)
        return out

    bound_opts = top_level(bound_lib, must_exist=True)
    distractor_opts = top_level(distractor_lib, must_exist=False)
    if not bound_opts or not distractor_opts:
        return None
    return rng.choice(distractor_opts), rng.choice(bound_opts)


def salience_note(stim_meta: dict) -> str:
    """An explicit cue stating the true binding, for the salience-intervention condition."""
    from .lexicons import IMPORT_NAMES

    alias = stim_meta["alias"]
    bound = IMPORT_NAMES.get(stim_meta["bound_target"], stim_meta["bound_target"])
    return f"Note: in this script the name `{alias}` is bound to the {bound} library."


def _prepend(note, user):
    return f"{note}\n\n{user}" if note else user


def forced_choice_messages(stim_meta: dict, distractor_method: str, bound_method: str, rng,
                           note: str | None = None):
    """Build the 2AFC chat messages with randomised option order.

    The distractor exists on the contrasting library but not on the bound one (a genuine
    AttributeError); the bound method exists on the bound library (the correct answer).
    Returns (messages, info) with the letter mapping. The caller decides whether the distractor
    is the prior library's method (the prior-pull case) or the other library's. ``note``, if
    given, prepends an explicit salience cue about the binding.
    """
    alias = stim_meta["alias"]
    code = context_code(stim_meta)
    distractor_opt = f"{alias}.{distractor_method}"
    bound_opt = f"{alias}.{bound_method}"
    swap = rng.random() < 0.5
    if swap:
        a_opt, b_opt = bound_opt, distractor_opt
        distractor_letter, bound_letter = "B", "A"
    else:
        a_opt, b_opt = distractor_opt, bound_opt
        distractor_letter, bound_letter = "A", "B"
    user = _prepend(note, (
        "Here is a Python script:\n\n"
        "```python\n" + code + "\n```\n\n"
        "Exactly one of these two lines runs without raising AttributeError. Which one?\n"
        f"A. {a_opt}\n"
        f"B. {b_opt}\n"
        "Reply with only the single letter A or B."
    ))
    info = {
        "distractor_letter": distractor_letter,
        "bound_letter": bound_letter,
        "option_a": a_opt,
        "option_b": b_opt,
        "distractor_method": distractor_method,
        "bound_method": bound_method,
    }
    return [{"role": "user", "content": user}], info


def generation_messages(stim_meta: dict, note: str | None = None):
    """Free completion of the line ending at ``alias.`` (output only the rest of the line)."""
    code = context_code(stim_meta)
    alias = stim_meta["alias"]
    user = _prepend(note, (
        "Complete the LAST line of this Python script. Output only the text that comes after "
        f"`{alias}.` to finish that one line, nothing else (no explanation, no code fence).\n\n"
        "```python\n" + code + "\n" + alias + ".\n```"
    ))
    return [{"role": "user", "content": user}]


def verbal_messages(stim_meta: dict, note: str | None = None):
    code = context_code(stim_meta)
    alias = stim_meta["alias"]
    user = _prepend(note, (
        "Read this Python script:\n\n"
        "```python\n" + code + "\n```\n\n"
        f"Which Python library is the name `{alias}` bound to here? "
        "Answer with only the library's import name (one word)."
    ))
    return [{"role": "user", "content": user}]


# --------------------------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------------------------
def parse_choice(result: ChatResult) -> str | None:
    """Extract the chosen letter (A/B) from content, falling back to the first logprob token.

    Matches a STANDALONE letter, so a word such as "answer" does not register as an "a". The
    instructed reply is a bare "A"/"B"; the regex also recovers "The answer is B." style replies.
    """
    import re

    for src in (result.content or "", result.first_token or ""):
        s = src.strip()
        if s.upper() in ("A", "B"):
            return s.upper()
        m = re.search(r"\b([ABab])\b", s)
        if m:
            return m.group(1).upper()
    return None


def chosen_logprob(result: ChatResult, letter: str) -> float | None:
    """Log-prob the model placed on the chosen letter, if exposed (confidence side-signal)."""
    if result.first_token and result.first_token.strip().upper().startswith(letter):
        if result.first_token_logprob is not None and result.first_token_logprob > _LOGPROB_FLOOR:
            return result.first_token_logprob
    for tok, lp in result.top_logprobs:
        if tok and tok.strip().upper() == letter and lp > _LOGPROB_FLOOR:
            return lp
    return None


def parse_attribute(generated: str, alias: str | None = None) -> str | None:
    """First identifier the completion accesses on the alias (e.g. ``array`` from ``array(...)``).

    Chat models wrap completions in prose and markdown, so before reading the identifier we strip
    a leading code fence (```python``), stray backticks, and a re-echoed ``alias.`` prefix. The
    free-form arm is still noisier than the controlled forced choice; callers should treat its
    rate as descriptive.
    """
    import re

    text = generated.strip()
    text = re.sub(r"^```[A-Za-z0-9_]*\s*", "", text)  # opening fence, optional language tag
    text = text.strip().lstrip("`").strip()
    if alias:
        # collapse a re-echoed "alias." (possibly repeated) so we read the method, not the alias
        prefix = f"{alias}."
        while text.startswith(prefix):
            text = text[len(prefix):]
    m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text)
    return m.group(0) if m else None


def match_library(answer: str, candidates: dict[str, str]) -> str | None:
    """Map a verbal answer to one of ``candidates`` (key -> import name); longest name first."""
    low = answer.lower()
    for key, name in sorted(candidates.items(), key=lambda kv: -len(kv[1])):
        if name.lower() in low:
            return key
    return None
