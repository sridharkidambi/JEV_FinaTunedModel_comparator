from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RouteResult:
    """Outcome of one system's routing decision for a single input."""

    system: str
    chosen_agent: Optional[str]
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    raw: Optional[dict] = None
