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
        self.world.populate(self.rng)
        self.environment = Environment(day_length=self.config.day_length)
        self.pheromone_field = PheromoneField(
            width=self.config.world_width,
            height=self.config.world_height,
        )
        self._apply_pheromone_config()

    def add_colony(self, colony: Colony) -> None:
        """Register a colony, mark its nest, and spawn initial ants.

        Args:
            colony: The colony to add.
        """
        self.world.mark_nest(colony.nest_x, colony.nest_y)
        for _ in range(self.config.initial_ants):
            colony.spawn_ant(self.rng)
        self.colonies.append(colony)

    def step(self) -> None:
        """Advance the simulation by one tick.

        Follows the canonical tick order:
        1. Environment
        2. Pheromones
        3. Ants
        4. Conflicts
        5. Colony-level effects (food, aging, starvation, brood, death)
        """
        from anthemyr.pheromones.fields import PheromoneType

        # 1. Update environment
        self.environment.update(self.world, self.rng)

        # 1b. Regenerate food on the world grid
        self.world.regenerate_food(
            self.rng,
            regen_rate=self.config.food_regen_rate,
            food_cap=self.config.food_cap,
        )

        # 2. Update pheromone fields
        update_field(self.pheromone_field)

        # 3. Update ants
        for colony in self.colonies:
            for ant in colony.ants:
                food = ant.update(
                    self.world,
                    self.pheromone_field,
                    colony.nest_x,
                    colony.nest_y,
                    self.rng,
                )
                colony.food_stores += food

        # 4. Resolve conflicts
        self._resolve_conflicts()

        # 5. Colony-level effects
        for colony in self.colonies:
            colony.consume_food(self.config.consumption_per_ant)
            colony.apply_food_pressure(
                self.config.comfort_food_per_ant,
                self.config.max_starvation_damage,
            )
            colony.apply_aging(self.config.max_age)

            # Deposit DEATH pheromone at corpse sites before removing
            dead = colony.remove_dead()
            for ant in dead:
                self.pheromone_field.deposit(
                    PheromoneType.DEATH,
                    ant.x,
                    ant.y,
                    5.0,
                )

            # Brood lifecycle
            colony.lay_eggs(
                self.config.egg_rate,
                self.config.comfort_food_per_ant,
                self.rng,
            )
            colony.develop_brood(
                self.config.brood_mature_ticks,
                self.rng,
            )

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

    def _apply_pheromone_config(self) -> None:
        """Apply per-type diffusion/evaporation rates from config."""
        from anthemyr.pheromones.fields import PheromoneType

        name_map: dict[str, PheromoneType] = {
            pt.name.lower(): pt for pt in PheromoneType
        }
        for name, rates in self.config.pheromone_defaults.items():
            ptype = name_map.get(name)
            if ptype is None:
                continue
            layer = self.pheromone_field.layers[ptype]
            if "diffusion_rate" in rates:
                layer.diffusion_rate = rates["diffusion_rate"]
            if "evaporation_rate" in rates:
                layer.evaporation_rate = rates["evaporation_rate"]
