"""Prompt moderation gate (content safety).

Runs BEFORE a job is admitted to the queue — rejected prompts never burn
GPU capacity. Two modes, composable:

* `blocklist`  — deterministic regex/keyword policy (default; zero-cost).
* `classifier` — pluggable scoring hook (e.g. NSFW text classifier or a
  managed moderation API). Registered via `set_classifier_hook`, kept
  behind a function boundary so the core never imports a specific vendor.

Post-generation re-checks of OUTPUT media are the worker's responsibility
(defence in depth: prompts can be benign while outputs are not).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..exceptions import SafetyError
from ..logging import get_logger

log = get_logger(__name__)

# Illustrative starter policy — operators extend/replace via config.
# Kept intentionally short here; production policy lives in config, not code.
_DEFAULT_PATTERNS: tuple[str, ...] = (
    r"\b(csam|child\s+(?:porn|sexual))\b",
    r"\b(?:how\s+to\s+(?:build|make))\s+(?:a\s+)?(?:bomb|explosive|nerve\s+agent)\b",
    r"\bdeep\s*fake\s+(?:nude|of\s+a\s+(?:child|minor))\b",
)


@dataclass(frozen=True)
class ModerationVerdict:
    allowed: bool
    reason: str = ""
    score: float | None = None


class ClassifierHook(Protocol):
    def __call__(self, prompt: str) -> ModerationVerdict: ...


class ModerationGate:
    """Pre-queue safety gate with deterministic and pluggable layers."""

    def __init__(
        self,
        enabled: bool = True,
        mode: str = "blocklist",
        max_prompt_chars: int = 2000,
        patterns: tuple[str, ...] | None = None,
    ) -> None:
        self.enabled = enabled
        self.mode = mode
        self.max_prompt_chars = max_prompt_chars
        self._patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or _DEFAULT_PATTERNS)]
        self._classifier: ClassifierHook | None = None
        self._rejected_total = 0

    def set_classifier_hook(self, hook: ClassifierHook) -> None:
        """Install the model/API-backed scorer (no vendor import in core)."""
        self._classifier = hook

    def check(self, prompt: str) -> ModerationVerdict:
        if not self.enabled:
            return ModerationVerdict(allowed=True)
        if len(prompt) > self.max_prompt_chars:
            self._rejected_total += 1
            return ModerationVerdict(allowed=False, reason="prompt exceeds max length")
        if self.mode in ("blocklist", "both"):
            for pattern in self._patterns:
                if pattern.search(prompt):
                    self._rejected_total += 1
                    log.warning(
                        "prompt rejected by policy",
                        extra={"rule": pattern.pattern[:48], "chars": len(prompt)},
                    )
                    return ModerationVerdict(allowed=False, reason="prompt matches blocked policy")
        if self.mode in ("classifier", "both") and self._classifier is not None:
            verdict = self._classifier(prompt)
            if not verdict.allowed:
                self._rejected_total += 1
                return verdict
        return ModerationVerdict(allowed=True)

    def enforce(self, prompt: str) -> None:
        """Raise SafetyError when the prompt must not be queued."""
        verdict = self.check(prompt)
        if not verdict.allowed:
            raise SafetyError(
                verdict.reason or "prompt rejected",
                details={
                    "score": verdict.score,
                },
            )

    def rejected_total(self) -> int:
        return self._rejected_total


def reject_score_above(threshold: float) -> Callable[[str], ModerationVerdict]:
    """Helper to build classifier hooks from a score function threshold."""

    def wrap(score_fn: Callable[[str], float]) -> ClassifierHook:
        def hook(prompt: str) -> ModerationVerdict:
            score = score_fn(prompt)
            if score > threshold:
                return ModerationVerdict(
                    allowed=False, reason="classifier score above threshold", score=score
                )
            return ModerationVerdict(allowed=True, score=score)

        return hook

    return wrap  # type: ignore[return-value]
