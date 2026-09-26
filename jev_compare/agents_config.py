from pathlib import Path

import yaml


def load_agents_config(path: str | Path) -> tuple[dict[str, str], str]:
    """Load the shared downstream-agent taxonomy used by every router.

    All three systems (JEV, the LoRA stand-in, and OpenRouter) are scored against
    the exact same agent names/descriptions and instructions so the
    comparison is apples-to-apples.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["agents"], data["instructions"]
