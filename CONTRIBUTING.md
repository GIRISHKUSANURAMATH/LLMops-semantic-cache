# Contributing

Thanks for your interest in improving this project.

## Setup

```bash
git clone https://github.com/GIRISHKUSANURAMATH/llmops-semantic-cache.git
cd llmops-semantic-cache
python -m venv .venv
```

Activate the environment:

| Shell | Command |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows CMD | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Then install with development tools:

```bash
python -m pip install -e ".[dev]"
```

## Checks (all must pass before a PR is merged)

```bash
ruff check .
ruff format --check .
mypy .                 # strict on src/, scripts/, examples/
pytest
```

`pytest` runs unit, integration and end-to-end tests. None of them need network access or an
API key. Run a subset with `pytest tests/unit`, `pytest -m integration` or `pytest -m e2e`.

## Guidelines

- Keep FastAPI routes thin; caching logic belongs in `llmops_cache/cache/service.py`.
- New embedding or LLM backends implement `EmbeddingProvider` / `LLMProvider`; new vector
  databases implement `VectorStore`. Add tests that use mocks, not real paid APIs.
- Anything that changes what a response depends on (a new generation parameter, tool config,
  etc.) must be added to the cache context key in `cache/identity.py`.
- Never commit `.env`, API keys or generated `data/` files.

## Process

1. Open an issue describing the bug or feature.
2. Create a branch, make focused commits, add or update tests.
3. Open a pull request; CI must be green.
