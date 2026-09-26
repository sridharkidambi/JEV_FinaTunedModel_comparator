import os
import time

from ..schemas import RouteResult

_DEFAULT_MODEL = os.environ.get("LORA_STUB_MODEL", "valhalla/distilbart-mnli-12-3")

# Rough keyword prior used only when transformers/torch aren't installed, so
# the harness still produces a result instead of hard-failing.
_KEYWORD_MAP = {
    "billing_support": ["invoice", "statement", "bill", "subscription"],
    "payment_failure": ["declined", "failed", "failure", "error", "retry"],
    "fraud_risk": ["fraud", "unauthorized", "suspicious", "chargeback", "dispute"],
    "refunds": ["refund", "cancel", "reverse", "money back"],
}


class LoraStubClient:
    """Stand-in for a LoRA-fine-tuned payments/billing small language model.

    Until a real LoRA adapter is trained, this uses an off-the-shelf
    zero-shot classifier (a small NLI model) scored against the same agent
    taxonomy as JEV and OpenRouter. If transformers/torch aren't installed, it
    falls back to a naive keyword prior and flags that in `error` so results
    aren't silently misread as the real model.

    To swap in a real LoRA adapter later, replace `_load`/`route` with a
    PEFT-loaded causal/sequence-classification model call that returns the
    same RouteResult shape.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or _DEFAULT_MODEL
        self._pipeline = None
        self._load_error: str | None = None

    def _load(self):
        if self._pipeline is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline("zero-shot-classification", model=self.model_name)
        except Exception as e:
            self._load_error = str(e)

    def route(self, text: str, agents: dict[str, str], instructions: str) -> RouteResult:
        self._load()
        agent_names = list(agents.keys())
        start = time.perf_counter()

        if self._pipeline is not None:
            try:
                result = self._pipeline(
                    text,
                    candidate_labels=agent_names,
                    hypothesis_template="This message should be routed to {}.",
                )
                latency_ms = (time.perf_counter() - start) * 1000
                probs = dict(zip(result["labels"], result["scores"]))
                chosen = result["labels"][0]
                return RouteResult(
                    system="lora_stub",
                    chosen_agent=chosen,
                    probabilities=probs,
                    confidence=probs[chosen],
                    latency_ms=latency_ms,
                    raw=result,
                )
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                return RouteResult(system="lora_stub", chosen_agent=None, latency_ms=latency_ms, error=str(e))

        probs = _keyword_baseline(text, agent_names)
        latency_ms = (time.perf_counter() - start) * 1000
        chosen = max(probs, key=probs.get)
        return RouteResult(
            system="lora_stub",
            chosen_agent=chosen,
            probabilities=probs,
            confidence=probs[chosen],
            latency_ms=latency_ms,
            error=f"transformers unavailable, used keyword fallback ({self._load_error})",
        )


def _keyword_baseline(text: str, agent_names: list[str]) -> dict[str, float]:
    text_lower = text.lower()
    scores = {name: 0 for name in agent_names}
    for name in agent_names:
        for kw in _KEYWORD_MAP.get(name, []):
            if kw in text_lower:
                scores[name] += 1

    total = sum(scores.values())
    if total == 0:
        n = len(agent_names)
        return {name: 1 / n for name in agent_names}
    return {name: score / total for name, score in scores.items()}
