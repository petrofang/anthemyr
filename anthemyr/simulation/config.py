"""Config — load simulation parameters from YAML files.

All tunable constants (world size, pheromone rates, species profiles,
trait schemas) live in YAML and are parsed into typed dataclasses here.
This keeps the simulation core data-driven and easy to experiment with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SimulationConfig:
    """Top-level simulation configuration.

    Attributes:
        seed: RNG seed for deterministic replay.
        world_width: Number of grid columns.
        world_height: Number of grid rows.
        day_length: Ticks per full day/night cycle.
        initial_ants: Starting ant population per colony.
        max_age: Maximum ant lifespan in ticks before death.
        comfort_food_per_ant: Food-per-ant level at which health
            pressure is zero and reproduction begins.
        max_starvation_damage: Maximum HP lost per ant per tick when
            food-per-ant is zero.
        consumption_per_ant: Food consumed per ant per tick.
        base_regen_rate: Probability of spontaneous food regrowth per
            empty cell per tick.
        spread_regen_rate: Probability scaling for adjacency-driven food
            spread from neighbouring cells.
        food_cap: Maximum food a cell can hold.
        egg_rate: Maximum eggs per tick at high food-per-ant.
        brood_mature_ticks: Ticks for brood to mature into an adult ant.
        pheromone_defaults: Per-type diffusion/evaporation overrides.
    """

    seed: int = 42
    world_width: int = 64
    world_height: int = 64
    day_length: int = 100
    initial_ants: int = 50

    # Ant lifecycle
    max_age: int = 1000
    comfort_food_per_ant: float = 1.0
    max_starvation_damage: float = 0.04
    consumption_per_ant: float = 0.02

    # Food regeneration (adjacency-based spread)
    base_regen_rate: float = 0.0005
    spread_regen_rate: float = 0.02
    food_cap: float = 5.0

    # Brood development
    egg_rate: float = 0.5
    brood_mature_ticks: int = 40

    pheromone_defaults: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimulationConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            A populated SimulationConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            seed=data.get("seed", cls.seed),
            world_width=data.get("world_width", cls.world_width),
            world_height=data.get("world_height", cls.world_height),
            day_length=data.get("day_length", cls.day_length),
            initial_ants=data.get("initial_ants", cls.initial_ants),
            max_age=data.get("max_age", cls.max_age),
            comfort_food_per_ant=data.get(
                "comfort_food_per_ant",
                cls.comfort_food_per_ant,
            ),
            max_starvation_damage=data.get(
                "max_starvation_damage",
                cls.max_starvation_damage,
            ),
            consumption_per_ant=data.get(
                "consumption_per_ant",
                cls.consumption_per_ant,
            ),
            base_regen_rate=data.get(
                "base_regen_rate",
                cls.base_regen_rate,
            ),
            spread_regen_rate=data.get(
                "spread_regen_rate",
                cls.spread_regen_rate,
            ),
            food_cap=data.get("food_cap", cls.food_cap),
            egg_rate=data.get("egg_rate", cls.egg_rate),
            brood_mature_ticks=data.get(
                "brood_mature_ticks",
                cls.brood_mature_ticks,
            ),
            pheromone_defaults=data.get("pheromone_defaults", {}),
        )
