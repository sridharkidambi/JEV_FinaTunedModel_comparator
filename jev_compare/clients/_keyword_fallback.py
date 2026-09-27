"""Naive keyword prior shared by local-model clients when their real
dependency (transformers, laya, ...) isn't installed, so the harness still
produces a result instead of hard-failing.
"""

_KEYWORD_MAP = {
    "billing_support": ["invoice", "statement", "bill", "subscription"],
    "payment_failure": ["declined", "failed", "failure", "error", "retry"],
    "fraud_risk": ["fraud", "unauthorized", "suspicious", "chargeback", "dispute"],
    "refunds": ["refund", "cancel", "reverse", "money back"],
}


def keyword_baseline(text: str, agent_names: list[str]) -> dict[str, float]:
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
