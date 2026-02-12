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
        pheromone_defaults: Per-type diffusion/evaporation overrides.
    """

    seed: int = 42
    world_width: int = 64
    world_height: int = 64
    day_length: int = 100
    initial_ants: int = 50
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
            pheromone_defaults=data.get("pheromone_defaults", {}),
        )
