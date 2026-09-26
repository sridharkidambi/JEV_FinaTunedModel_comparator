# JEV compare

Given a payments/billing message, ask three systems to decide which
downstream agent should handle it, and compare their chosen agent,
confidence/probability distribution, and latency side by side:

1. **JEV** — TypeSafe AI's "System One" model (`jev-latest`), called via its
   real `POST https://api.typesafe.ai/v1/systemone` API with a `choice`
   question. Returns calibrated probabilities natively.
   ([announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
   [API reference](https://docs.typesafe.ai/api))
2. **`lora_stub`** — stand-in for a LoRA-fine-tuned small payments/billing
   model. No adapter has been trained yet, so this currently runs an
   off-the-shelf zero-shot classifier (`valhalla/distilbart-mnli-12-3` by
   default) against the same agent taxonomy. If `transformers`/`torch`
   aren't installed it falls back to a naive keyword baseline and says so in
   the result's `error` field, so a placeholder result is never mistaken for
   the real model.
3. **OpenRouter** — frontier-model baseline via OpenRouter's OpenAI-compatible
   Chat Completions API (default model `openai/gpt-5`, but any model
   OpenRouter hosts can be swapped in via `--openrouter-model`), forced
   through tool-use to return the same shape (chosen agent, self-estimated
   probability distribution, confidence).

All three are scored against the identical agent list/instructions in
[`config/agents.yaml`](config/agents.yaml), run **concurrently** (each times
itself independently, so wall-clock cost stays low while per-system latency
stays accurate).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in TYPESAFE_API_KEY and OPENROUTER_API_KEY
```

For real zero-shot inference in the `lora_stub` client, also install:

```bash
pip install transformers torch
```

(Without these, the harness still runs, using the keyword fallback.)

## Usage

```bash
python -m jev_compare.cli "My card was charged twice and I never authorized it." --save results.json
```

Or pipe text in:

```bash
echo "Our webhook integration keeps returning 500s on refund events." | python -m jev_compare.cli
```

This prints a table like:

| System | Chosen agent | Confidence | Latency (ms) | Top probabilities | Notes |
|---|---|---|---|---|---|
| jev | fraud_risk | 0.81 | 210 | fraud_risk=0.81, billing_support=0.12 | |
| lora_stub | fraud_risk | 0.64 | 340 | fraud_risk=0.64, billing_support=0.28 | |
| openrouter | fraud_risk | 0.9 | 1450 | fraud_risk=0.90, billing_support=0.08 | |

`--save` writes the full `RouteResult` list (including raw provider
responses) as JSON for later analysis.

## Changing the agent taxonomy

Edit [`config/agents.yaml`](config/agents.yaml) — `instructions` is the
routing prompt, `agents` maps each downstream agent's name to a short
description. All three systems automatically pick up any change, since the
harness passes the same config to each client.

## Replacing the LoRA stand-in with a real adapter

Once a LoRA adapter is trained on payments/billing routing data, replace
`_load`/`route` in
[`jev_compare/clients/lora_stub_client.py`](jev_compare/clients/lora_stub_client.py)
with PEFT-loaded inference (base model + adapter) that returns a
`RouteResult` with the same shape — no changes needed elsewhere.

## Tests

```bash
pytest
```

Tests mock all three routers, so they run without network access or
`transformers`/`torch` installed.
