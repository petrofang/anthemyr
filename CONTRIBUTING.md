# Contributing to Anthemyr

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
```

## Dev commands

| Task         | Command                        |
|--------------|--------------------------------|
| Run tests    | `pytest -v`                    |
| Lint         | `ruff check anthemyr/ tests/`  |
| Auto-format  | `ruff format .`                |
| All hooks    | `pre-commit run --all-files`   |

## Coding conventions

- **Python 3.10+** — use type hints, dataclasses, `match` where appropriate.
- **Docstrings:** Google style on all public classes and functions.
- **Formatting & linting:** Ruff (configured in `pyproject.toml`).
- **Tests:** pytest. Mirror the source tree under `tests/` (e.g. `tests/test_ant.py` for `anthemyr/colony/ant.py`).
- **Determinism:** Always use seeded NumPy `Generator` instances — never bare `random.*`.

## Branching

- Branch from `main` with `feature/…`, `fix/…`, or `docs/…` prefixes.
- Include tests for new logic; ensure `pytest` and `ruff check` pass before pushing.

## Pre-commit hooks

Hooks run automatically:
- **On commit:** Ruff lint + format checks, trailing whitespace, YAML validation.
- **On push:** Full `pytest` suite.

If a hook fails, fix the issue and re-stage. Run `pre-commit run --all-files` to check everything manually.

## Running the simulation (dev example)

```python
from anthemyr.simulation.config import SimulationConfig
from anthemyr.simulation.engine import SimulationEngine

cfg = SimulationConfig.from_yaml("config/default.yaml")
engine = SimulationEngine(config=cfg)
engine.run(ticks=10)
```

## Questions

Open an issue or comment on the PR — happy to help.
