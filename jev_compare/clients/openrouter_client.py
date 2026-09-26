import json
import os
import time

from ..schemas import RouteResult

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Frontier-model baseline: any model via OpenRouter's Chat Completions API.

    Forces a tool call so the response is structured the same way as the
    other two systems (a chosen agent, a self-estimated probability
    distribution over all candidates, and an overall confidence).
    """

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-5"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=OPENROUTER_BASE_URL)
        return self._client

    def route(self, text: str, agents: dict[str, str], instructions: str) -> RouteResult:
        if not self.api_key:
            return RouteResult(system="openrouter", chosen_agent=None, error="OPENROUTER_API_KEY not set")

        agent_names = list(agents.keys())
        tool = {
            "type": "function",
            "function": {
                "name": "route_ticket",
                "description": instructions,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string", "enum": agent_names},
                        "probabilities": {
                            "type": "object",
                            "description": "Estimated probability for every candidate agent; should sum to ~1.0",
                            "properties": {name: {"type": "number"} for name in agent_names},
                            "required": agent_names,
                        },
                        "confidence": {"type": "number", "description": "Overall confidence in the chosen agent, 0-1"},
                    },
                    "required": ["agent", "probabilities", "confidence"],
                },
            },
        }
        agent_desc = "\n".join(f"- {name}: {desc}" for name, desc in agents.items())
        prompt = f"{instructions}\n\nCandidate downstream agents:\n{agent_desc}\n\nMessage:\n{text}"

        start = time.perf_counter()
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": "route_ticket"}},
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = (time.perf_counter() - start) * 1000
            tool_call = resp.choices[0].message.tool_calls[0]
            data = json.loads(tool_call.function.arguments)
            tokens_used = resp.usage.total_tokens if resp.usage else None
            return RouteResult(
                system="openrouter",
                chosen_agent=data.get("agent"),
                probabilities=data.get("probabilities", {}),
                confidence=data.get("confidence"),
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                raw=data,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return RouteResult(system="openrouter", chosen_agent=None, latency_ms=latency_ms, error=str(e))
