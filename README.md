# Representational geometry and in-context generalization in LLMs

Final project for **Neuro 240 / Phys 240: Biological and Artificial Intelligence** (Fall 2026), based on course Project 5.

**Question:** can the geometry of an LLM's internal representations predict how well it learns a categorization from a few examples in the prompt?

- **Exp 1 (core):** apply the few-shot learning theory of Sorscher, Ganguli & Sompolinsky (2022, *PNAS*), developed for macaque IT and vision networks, to LLM representations, and compare its predictions with in-context accuracy.
- **Exp 2:** when the demonstrations regroup the *same* items by a different attribute, do representations shift toward the demonstrated grouping (and where), and does the shift predict accuracy on unseen items?
- **Exp 3:** use Anthropic's Jacobian lens to read out the rule the model infers, and test whether "J-space" geometry predicts generalization better than full hidden states.

## Status

Experimental infrastructure built and smoke-tested end to end; no experiments have been run yet. How to run everything: [`docs/RUNBOOK.md`](docs/RUNBOOK.md). See [`tickets/BOARD.md`](tickets/BOARD.md).

## Repo guide

| Path | What |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | Full research and engineering plan |
| [`CLAUDE.md`](CLAUDE.md) | Context and rules for agents (and a good human overview) |
| [`docs/course/COURSE_CONTEXT.md`](docs/course/COURSE_CONTEXT.md) | Course, rules, original Project 5 brief |
| [`docs/background/`](docs/background/) | Theory, Jacobian lens, neuroscience connections |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Decision log |
| [`tickets/`](tickets/) | Ticket system ([how it works](tickets/README.md)) |

## Tickets quickstart

```bash
python3 scripts/tickets.py ready                    # what can be started now
python3 scripts/tickets.py show T-002               # read a ticket
python3 scripts/tickets.py set T-002 status in-progress owner yashwanth
python3 scripts/tickets.py list --phase 1           # filter by phase/status/owner/priority
python3 scripts/tickets.py new "Title" --phase 3 --depends T-031
python3 scripts/tickets.py board                    # regenerate tickets/BOARD.md
python3 scripts/tickets.py check                    # validate ids, deps, cycles
```

## Environment setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create .venv from pyproject.toml / uv.lock
uv run pytest                             # unit tests (add -m "not slow" to skip model-loading tests)
uv run ruff check .                       # lint
uv run python -c "import icl_geometry"    # sanity check
```

Hardware notes (Apple Silicon, MPS vs CPU; model sizes; throughput) are in [`docs/COMPUTE.md`](docs/COMPUTE.md). Model weights download to the Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`); never commit them.
