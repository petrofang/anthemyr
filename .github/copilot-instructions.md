# Anthemyr — Copilot Instructions

## What is this project?

Anthemyr is a science-driven ant superorganism MMO simulator (Python). The colony is the player character — individual ants follow local rules (response thresholds, pheromone gradients) producing emergent global behavior. Players influence the colony via policies and genetics, not direct unit control.

## Current status

**Pre-code / greenfield.** `README.md` is the authoritative design spec. Follow the proposed structure and phased roadmap when scaffolding.

## Project structure (target)

```
anthemyr/              # Package root
  world/               # World grid, cells, environment (weather, predators)
  colony/              # Colony state, Ant agents, traits, policies, evolution
  pheromones/           # Multi-layer pheromone fields, diffusion/evaporation
  simulation/           # Engine (tick loop), config loading (YAML)
  ui/                   # Phase 2 — Pygame 2D visualization
  net/                  # Phase 3 — Authoritative server, protocol, client API
  experiments/          # Jupyter notebooks and one-off prototypes
tests/                 # Mirrors anthemyr/ — one test file per module
```

Keep the simulation core **engine-agnostic** — no UI or network imports in `world/`, `colony/`, `pheromones/`, or `simulation/`.

## Key design patterns

- **Agent-based simulation:** Each `Ant` holds internal response thresholds and makes local decisions. Task allocation emerges from threshold distributions, not explicit assignments.
- **Multi-layer pheromone fields:** Trail, alarm, territory, recruitment, brood-care, death, royal — each with independent diffusion and evaporation rates. Implement as separate grid layers in `pheromones/fields.py`.
- **Policy-driven control:** Players set colony-wide policy sliders (aggression, exploration, sanitation, caste ratios) that shift threshold distributions. Policies live in `colony/policies.py`; traits in `colony/traits.py`.
- **Deterministic simulation:** Given a seed, the tick loop must be fully reproducible for debugging and replay. Use seeded RNG everywhere — never bare `random.*` calls.
- **Config-driven:** Species, world parameters, and trait schemas are loaded from **YAML** via `simulation/config.py`.

## Simulation tick order

Follow this exact sequence in `simulation/engine.py`:

1. Update environment (weather, day/night, predator spawns)
2. Update pheromone fields (diffuse, evaporate)
3. Update ants (local decisions, movement, interactions)
4. Resolve conflicts (combat, deaths, disease)
5. Apply colony-level effects (food consumption, brood development)

## Coding conventions

- **Python 3.10+** — use type hints, dataclasses, and `match` statements where appropriate.
- **Packaging:** `pyproject.toml` is the single source for project metadata and dependencies (no `requirements.txt`).
- **Linting & formatting:** Use **Ruff** for both linting and formatting. Configure in `pyproject.toml` under `[tool.ruff]`.
- **Docstrings:** Google style. All public classes and functions must have docstrings.
- **Testing:** Use **pytest**. Tests live in `tests/` mirroring the source tree (e.g., `tests/test_ant.py` for `anthemyr/colony/ant.py`). Run with `pytest` from the project root. Simulation determinism is a first-class testable property.
- Name modules and classes to match the proposed structure in README.md (e.g., `World`, `Colony`, `Ant`, `PheromoneField`).
- Instrument with metrics (food intake, mortality, brood survival) — these feed the policy dashboard and generational evolution system.

## What NOT to do

- Don't couple simulation logic to rendering or networking — those are later phases.
- Don't model ants as directly player-commanded units — they follow local rules only.
- Don't use global mutable state for RNG — pass seeded generators explicitly.
