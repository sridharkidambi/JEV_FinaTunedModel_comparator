import concurrent.futures as cf
from typing import Protocol

from .schemas import RouteResult


class Router(Protocol):
    def route(self, text: str, agents: dict[str, str], instructions: str) -> RouteResult: ...


class ComparisonHarness:
    """Runs the same text through every registered router concurrently.

    Each router times its own call internally, so `latency_ms` on the
    returned RouteResult reflects that system's real cost even though all
    three run in parallel to keep total wall-clock time low.
    """

    def __init__(self, clients: dict[str, Router], agents: dict[str, str], instructions: str):
        self.clients = clients
        self.agents = agents
        self.instructions = instructions

    def run(self, text: str) -> list[RouteResult]:
        results: dict[str, RouteResult] = {}
        with cf.ThreadPoolExecutor(max_workers=len(self.clients)) as ex:
            futures = {
                ex.submit(client.route, text, self.agents, self.instructions): name
                for name, client in self.clients.items()
            }
            for fut in cf.as_completed(futures):
                name = futures[fut]
                results[name] = fut.result()
        return [results[name] for name in self.clients]
