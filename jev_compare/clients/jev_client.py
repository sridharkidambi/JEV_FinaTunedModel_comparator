import os
import time

import httpx

from ..schemas import RouteResult

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class JevClient:
    """Client for TypeSafe AI's Jev ("System One") model.

    Jev takes unstructured state plus a typed `choice` question and returns
    calibrated probabilities, a confidence score, and the winning choice —
    see https://docs.typesafe.ai/api and
    https://typesafe.ai/blog/introducing-system-one-models-and-jev
    """

    def __init__(self, api_key: str | None = None, model: str = "jev-latest", timeout: float = 10.0):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.model = model
        self.timeout = timeout

    def route(self, text: str, agents: dict[str, str], instructions: str) -> RouteResult:
        if not self.api_key:
            return RouteResult(system="jev", chosen_agent=None, error="TYPESAFE_API_KEY not set")

        payload = {
            "state": text,
            "model": self.model,
            "questions": {
                "agent": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": dict(agents),
                }
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        start = time.perf_counter()
        try:
            resp = httpx.post(JEV_ENDPOINT, json=payload, headers=headers, timeout=self.timeout)
            latency_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            data = resp.json()
            answer = data["answers"]["agent"]
            usage = data.get("usage") or {}
            tokens_used = None
            if "input_tokens" in usage or "output_tokens" in usage:
                tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            return RouteResult(
                system="jev",
                chosen_agent=answer.get("choice"),
                probabilities=answer.get("probabilities", {}),
                confidence=answer.get("confidence"),
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                raw=data,
            )
        except Exception as e:  # network error, bad status, malformed body, etc.
            latency_ms = (time.perf_counter() - start) * 1000
            return RouteResult(system="jev", chosen_agent=None, latency_ms=latency_ms, error=str(e))
