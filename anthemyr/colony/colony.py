"""Colony — aggregate state for one ant colony.

A Colony owns its population of Ant agents, food stores, brood, queen
state, genetic profile, and policy parameters.  It is updated once per
tick after individual ants have acted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

from anthemyr.colony.ant import Ant
from anthemyr.colony.policies import Policies
from anthemyr.colony.traits import Traits


@dataclass
class Colony:
    """Top-level state for a single ant colony.

    Attributes:
        colony_id: Unique identifier.
        nest_x: X coordinate of the nest entrance.
        nest_y: Y coordinate of the nest entrance.
        food_stores: Accumulated food reserves.
        brood_count: Number of developing larvae/pupae.
        traits: Genetic trait profile for this colony.
        policies: Current policy-slider settings.
        ants: Living ant population.
        generation: Generational counter for evolutionary tracking.
    """

    colony_id: int
    nest_x: int
    nest_y: int
    food_stores: float = 100.0
    brood_count: int = 0
    traits: Traits = field(default_factory=Traits)
    policies: Policies = field(default_factory=Policies)
    ants: list[Ant] = field(default_factory=list)
    generation: int = 0

    def spawn_ant(self, rng: Generator) -> Ant:
        """Create a new ant at the nest with trait-derived thresholds.

        Args:
            rng: Seeded random generator.

        Returns:
            The newly created Ant (also appended to ``self.ants``).
        """
        ant = Ant.from_traits(
            x=self.nest_x,
            y=self.nest_y,
            traits=self.traits,
            rng=rng,
        )
        self.ants.append(ant)
        return ant

    def consume_food(self, amount_per_ant: float = 0.1) -> None:
        """Deduct per-tick food consumption for all living ants.

        Args:
            amount_per_ant: Food consumed per ant per tick.
        """
        consumption = len(self.ants) * amount_per_ant
        self.food_stores = max(0.0, self.food_stores - consumption)

    def remove_dead(self) -> list[Ant]:
        """Remove and return ants that have died (hp <= 0 or starved).

        Returns:
            List of ants that were removed.
        """
        dead = [a for a in self.ants if not a.is_alive]
        self.ants = [a for a in self.ants if a.is_alive]
        return dead
