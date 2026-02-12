"""SimulationEngine — the main tick loop.

Owns all top-level simulation state and advances it in the canonical
tick order defined in the design spec:

1. Update environment (weather, day/night, predator spawns)
2. Update pheromone fields (diffuse, evaporate)
3. Update ants (local decisions, movement, interactions)
4. Resolve conflicts (combat, deaths, disease)
5. Apply colony-level effects (food consumption, brood development)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator

from anthemyr.colony.colony import Colony
from anthemyr.pheromones.diffusion import update_field
from anthemyr.pheromones.fields import PheromoneField
from anthemyr.simulation.config import SimulationConfig
from anthemyr.world.environment import Environment
from anthemyr.world.world import World


@dataclass
class SimulationEngine:
    """Drives the simulation forward tick by tick.

    Attributes:
        config: Loaded simulation configuration.
        world: The spatial grid.
        environment: Global environmental state.
        pheromone_field: Multi-layer pheromone grids.
        colonies: All active colonies in the world.
        rng: Master seeded random generator.
        tick: Current tick count.
    """

    config: SimulationConfig
    world: World = field(init=False)
    environment: Environment = field(init=False)
    pheromone_field: PheromoneField = field(init=False)
    colonies: list[Colony] = field(init=False, default_factory=list)
    rng: Generator = field(init=False)
    tick: int = 0

    def __post_init__(self) -> None:
        """Build world, pheromone field, and RNG from config."""
        self.rng = np.random.default_rng(self.config.seed)
        self.world = World(
            width=self.config.world_width,
            height=self.config.world_height,
        )
        self.environment = Environment(day_length=self.config.day_length)
        self.pheromone_field = PheromoneField(
            width=self.config.world_width,
            height=self.config.world_height,
        )

    def add_colony(self, colony: Colony) -> None:
        """Register a colony in the simulation.

        Args:
            colony: The colony to add.
        """
        self.colonies.append(colony)

    def step(self) -> None:
        """Advance the simulation by one tick.

        Follows the canonical tick order:
        1. Environment
        2. Pheromones
        3. Ants
        4. Conflicts
        5. Colony-level effects
        """
        # 1. Update environment
        self.environment.update(self.world, self.rng)

        # 2. Update pheromone fields
        update_field(self.pheromone_field)

        # 3. Update ants
        for colony in self.colonies:
            for ant in colony.ants:
                ant.update(self.world, self.rng)

        # 4. Resolve conflicts
        self._resolve_conflicts()

        # 5. Colony-level effects
        for colony in self.colonies:
            colony.consume_food()
            colony.remove_dead()

        self.tick += 1

    def run(self, ticks: int) -> None:
        """Run the simulation for a fixed number of ticks.

        Args:
            ticks: Number of ticks to advance.
        """
        for _ in range(ticks):
            self.step()

    def _resolve_conflicts(self) -> None:
        """Handle combat, deaths, and disease across all colonies.

        TODO: Implement spatial conflict resolution.
        """
