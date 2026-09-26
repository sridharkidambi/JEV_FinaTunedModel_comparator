import time

from jev_compare.agents_config import load_agents_config
from jev_compare.cli import DEFAULT_CONFIG, build_table
from jev_compare.harness import ComparisonHarness
from jev_compare.schemas import RouteResult


class FakeRouter:
    """Deterministic stand-in for a real client, used to test the harness
    without hitting any network or requiring transformers/torch."""

    def __init__(self, name: str, delay: float = 0.0, chosen: str = "billing_support", error: str | None = None):
        self.name = name
        self.delay = delay
        self.chosen = chosen
        self.error = error

    def route(self, text, agents, instructions) -> RouteResult:
        start = time.perf_counter()
        if self.delay:
            time.sleep(self.delay)
        latency_ms = (time.perf_counter() - start) * 1000
        if self.error:
            return RouteResult(system=self.name, chosen_agent=None, latency_ms=latency_ms, error=self.error)
        probs = {name: 0.0 for name in agents}
        probs[self.chosen] = 0.9
        return RouteResult(
            system=self.name,
            chosen_agent=self.chosen,
            probabilities=probs,
            confidence=0.9,
            latency_ms=latency_ms,
        )


def test_load_agents_config():
    agents, instructions = load_agents_config(DEFAULT_CONFIG)
    assert "billing_support" in agents
    assert "fraud_risk" in agents
    assert instructions.strip()


def test_harness_runs_all_clients_and_preserves_order():
    agents, instructions = load_agents_config(DEFAULT_CONFIG)
    clients = {
        "jev": FakeRouter("jev", chosen="fraud_risk"),
        "lora_stub": FakeRouter("lora_stub", chosen="billing_support"),
        "openrouter": FakeRouter("openrouter", chosen="fraud_risk"),
    }
    harness = ComparisonHarness(clients, agents, instructions)
    results = harness.run("My card was charged twice without authorization.")

    assert [r.system for r in results] == ["jev", "lora_stub", "openrouter"]
    assert results[0].chosen_agent == "fraud_risk"
    assert results[1].chosen_agent == "billing_support"
    assert all(r.latency_ms is not None and r.latency_ms >= 0 for r in results)


def test_harness_runs_concurrently_not_sequentially():
    agents, instructions = load_agents_config(DEFAULT_CONFIG)
    clients = {name: FakeRouter(name, delay=0.2) for name in ("jev", "lora_stub", "openrouter")}
    harness = ComparisonHarness(clients, agents, instructions)

    start = time.perf_counter()
    harness.run("test")
    wall_time = time.perf_counter() - start

    # Three 0.2s calls sequentially would take ~0.6s; concurrently, well under that.
    assert wall_time < 0.5


def test_harness_surfaces_per_client_errors_without_failing_others():
    agents, instructions = load_agents_config(DEFAULT_CONFIG)
    clients = {
        "jev": FakeRouter("jev", error="TYPESAFE_API_KEY not set"),
        "lora_stub": FakeRouter("lora_stub", chosen="payment_failure"),
        "openrouter": FakeRouter("openrouter", chosen="payment_failure"),
    }
    harness = ComparisonHarness(clients, agents, instructions)
    results = harness.run("My payment keeps getting declined on retry.")

    jev_result = next(r for r in results if r.system == "jev")
    assert jev_result.error == "TYPESAFE_API_KEY not set"
    assert jev_result.chosen_agent is None

    others = [r for r in results if r.system != "jev"]
    assert all(r.chosen_agent == "payment_failure" for r in others)


def test_build_table_does_not_raise_on_missing_fields():
    results = [
        RouteResult(system="jev", chosen_agent=None, error="no api key"),
        RouteResult(system="openrouter", chosen_agent="refunds", probabilities={"refunds": 0.7}, confidence=0.7, latency_ms=812.3),
    ]
    table = build_table(results)
    assert table.row_count == 2
