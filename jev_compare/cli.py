import argparse
import json
import sys
from pathlib import Path

from .agents_config import load_agents_config
from .clients.openrouter_client import OpenRouterClient
from .clients.jev_client import JevClient
from .clients.lora_stub_client import LoraStubClient
from .harness import ComparisonHarness
from .schemas import RouteResult

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "agents.yaml"


def build_table(results: list[RouteResult]):
    from rich.table import Table

    table = Table(title="Routing comparison", header_style="bold white on dark_blue", show_lines=True, expand=True)
    table.add_column("System", style="bold cyan", no_wrap=True)
    table.add_column("Chosen agent", style="bold green")
    table.add_column("Confidence", style="yellow", justify="right")
    table.add_column("Latency (ms)", style="magenta", justify="right")
    table.add_column("Tokens", style="blue", justify="right")
    table.add_column("Top probabilities", style="white")
    table.add_column("Notes", style="bold red", overflow="fold")

    for r in results:
        top_probs = sorted(r.probabilities.items(), key=lambda kv: -kv[1])[:3]
        top = ", ".join(f"{k}={v:.2f}" for k, v in top_probs)
        table.add_row(
            r.system,
            r.chosen_agent or "-",
            f"{r.confidence:.2f}" if r.confidence is not None else "-",
            f"{r.latency_ms:.0f}" if r.latency_ms is not None else "-",
            f"{r.tokens_used:,}" if r.tokens_used is not None else "-",
            top or "-",
            r.error or "",
        )
    return table


def main(argv: list[str] | None = None):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Compare JEV, a LoRA-payments-SLM stand-in, and OpenRouter on a payments/billing routing decision."
    )
    parser.add_argument("text", nargs="?", help="Input text. If omitted, reads from stdin.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the shared agent taxonomy YAML.")
    parser.add_argument("--save", help="Path to save full results as JSON.")
    parser.add_argument("--openrouter-model", default="openai/gpt-5")
    args = parser.parse_args(argv)

    text = args.text or sys.stdin.read()
    if not text.strip():
        parser.error("No input text provided.")

    agents, instructions = load_agents_config(args.config)

    clients = {
        "jev": JevClient(),
        "lora_stub": LoraStubClient(),
        "openrouter": OpenRouterClient(model=args.openrouter_model),
    }
    harness = ComparisonHarness(clients, agents, instructions)
    results = harness.run(text)

    from rich.console import Console

    console = Console()
    console.print(build_table(results))

    if args.save:
        payload = [r.__dict__ for r in results]
        Path(args.save).write_text(json.dumps(payload, indent=2, default=str))
        console.print(f"Saved results to {args.save}")


if __name__ == "__main__":
    main()
