import os
import time

from ..schemas import RouteResult
from ._keyword_fallback import keyword_baseline

_DEFAULT_MODEL = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")

# Candidate keys for the full per-option distribution in a `predict()` answer.
# The published docs confirm a distribution is returned but not its exact key,
# so we check a few and fall back to just the winning choice if none match.
_DISTRIBUTION_KEYS = ("probabilities", "scores", "distribution")


class LayaClient:
    """Open-weight local equivalent of a "System One" model: Laya is a
    non-autoregressive decision engine built on a fine-tuned ModernBERT-large
    encoder plus a small decision head. Given state and a typed `choice`
    question it scores every candidate option in a single forward pass and
    returns calibrated probabilities, a confidence score, and the winning
    choice — the same shape Jev returns, but self-hosted.
    See https://huggingface.co/convaiinnovations/laya

    If the `laya` package isn't installed, this falls back to the same naive
    keyword prior lora_stub_client uses and flags that in `error`, so a
    placeholder result is never mistaken for the real model.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or _DEFAULT_MODEL
        self._agent = None
        self._load_error: str | None = None

    def _load(self):
        if self._agent is not None or self._load_error is not None:
            return
        try:
            import laya

            self._agent = laya.load(self.model_name)
        except Exception as e:
            self._load_error = str(e)

    def route(self, text: str, agents: dict[str, str], instructions: str) -> RouteResult:
        self._load()
        agent_names = list(agents.keys())

        if self._agent is not None:
            questions = {
                "agent": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": dict(agents),
                }
            }
            start = time.perf_counter()
            try:
                result = self._agent.predict(text, questions)
                latency_ms = (time.perf_counter() - start) * 1000
                answer = result["answers"]["agent"]
                chosen = answer.get("choice")
                confidence = answer.get("confidence")
                probs = next((answer[k] for k in _DISTRIBUTION_KEYS if k in answer), None)
                if probs is None:
                    probs = {chosen: confidence} if chosen is not None else {}
                usage = result.get("usage") or {}
                tokens_used = usage.get("input_tokens")
                return RouteResult(
                    system="laya",
                    chosen_agent=chosen,
                    probabilities=probs,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    tokens_used=tokens_used,
                    raw=result,
                )
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                return RouteResult(system="laya", chosen_agent=None, latency_ms=latency_ms, error=str(e))

        start = time.perf_counter()
        probs = keyword_baseline(text, agent_names)
        latency_ms = (time.perf_counter() - start) * 1000
        chosen = max(probs, key=probs.get)
        return RouteResult(
            system="laya",
            chosen_agent=chosen,
            probabilities=probs,
            confidence=probs[chosen],
            latency_ms=latency_ms,
            error=f"laya package unavailable, used keyword fallback ({self._load_error})",
        )
