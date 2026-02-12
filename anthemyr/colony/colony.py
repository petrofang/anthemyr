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
        brood_progress: Ticks accumulated toward next brood maturation.
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
    brood_progress: int = 0
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

    def apply_starvation(self, damage: float) -> None:
        """Damage all ants when the colony has no food.

        Args:
            damage: HP lost per ant this tick.
        """
        if self.food_stores > 0:
            return
        for ant in self.ants:
            ant.hp -= damage

    def apply_aging(self, max_age: int) -> None:
        """Kill ants that have exceeded their maximum lifespan.

        Args:
            max_age: Age in ticks after which an ant dies.
        """
        for ant in self.ants:
            if ant.age >= max_age:
                ant.hp = 0.0

    def lay_eggs(self, egg_rate: float, rng: Generator) -> None:
        """Queen lays eggs proportional to surplus food stores.

        Egg-laying costs food: each egg costs 1 unit.

        Args:
            egg_rate: Eggs per tick per unit of surplus food.
            rng: Seeded random generator.
        """
        if self.food_stores <= 10.0:
            return  # need a baseline before investing in brood

        surplus = self.food_stores - 10.0
        expected_eggs = surplus * egg_rate
        # Stochastic: fractional part becomes a probability
        whole = int(expected_eggs)
        frac = expected_eggs - whole
        eggs = whole + (1 if rng.random() < frac else 0)

        # Cap eggs by available food (each egg costs 1 unit)
        eggs = min(eggs, int(self.food_stores))
        if eggs > 0:
            self.brood_count += eggs
            self.food_stores -= eggs

    def develop_brood(
        self,
        mature_ticks: int,
        rng: Generator,
    ) -> int:
        """Advance brood development; mature brood become new ants.

        Brood progress accumulates each tick. When it reaches the
        maturation threshold, one brood hatches into an adult ant.

        Args:
            mature_ticks: Ticks required for one brood to mature.
            rng: Seeded random generator.

        Returns:
            Number of new ants hatched this tick.
        """
        if self.brood_count <= 0:
            return 0

        self.brood_progress += self.brood_count
        hatched = 0
        while self.brood_progress >= mature_ticks and self.brood_count > 0:
            self.brood_progress -= mature_ticks
            self.brood_count -= 1
            self.spawn_ant(rng)
            hatched += 1
        return hatched

    def remove_dead(self) -> list[Ant]:
        """Remove and return ants that have died (hp <= 0 or starved).

        Returns:
            List of ants that were removed.
        """
        dead = [a for a in self.ants if not a.is_alive]
        self.ants = [a for a in self.ants if a.is_alive]
        return dead
