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
    food_stores: float = 300.0
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

    def apply_food_pressure(
        self,
        comfort_food_per_ant: float,
        max_damage: float,
    ) -> None:
        """Apply smooth, density-dependent health pressure.

        When food-per-ant is at or above the comfort level, no damage.
        When food-per-ant is zero, each ant receives ``max_damage``.
        Between these extremes the damage scales linearly, creating a
        smooth negative-feedback loop: larger populations feel more
        pressure, naturally limiting growth.

        Args:
            comfort_food_per_ant: Food-per-ant level at which pressure
                is zero (colony is comfortable).
            max_damage: Maximum HP lost per ant per tick (at zero food).
        """
        n = len(self.ants)
        if n == 0:
            return

        food_per_ant = self.food_stores / n

        if food_per_ant >= comfort_food_per_ant:
            return  # colony is comfortable — no pressure

        # Linear interpolation: 0 food → max_damage, comfort → 0
        ratio = food_per_ant / comfort_food_per_ant
        damage = max_damage * (1.0 - ratio)

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

    def lay_eggs(
        self,
        egg_rate: float,
        comfort_food_per_ant: float,
        rng: Generator,
    ) -> None:
        """Queen lays eggs when per-capita food exceeds comfort level.

        Egg-laying scales with how far above comfort the colony is,
        creating a natural cap: as population grows, food-per-ant drops
        and reproduction slows — a smooth density-dependent brake.

        Below comfort but above 50% comfort, a trickle of eggs (10%
        of normal rate) are still laid, preventing total reproductive
        stall during mild stress.

        Each egg costs 1 food unit.

        Args:
            egg_rate: Maximum eggs per tick at very high food-per-ant.
            comfort_food_per_ant: Food-per-ant threshold below which
                reproduction slows dramatically.
            rng: Seeded random generator.
        """
        n = len(self.ants)
        # Need at least 1 ant (the queen) and some food
        if n == 0 or self.food_stores < 1.0:
            return

        food_per_ant = self.food_stores / n

        if food_per_ant <= comfort_food_per_ant * 0.5:
            return  # colony is severely stressed — no reproduction

        if food_per_ant <= comfort_food_per_ant:
            # Trickle reproduction: 10% rate when between 50%-100% comfort
            stress_ratio = (food_per_ant - comfort_food_per_ant * 0.5) / (
                comfort_food_per_ant * 0.5
            )
            expected_eggs = egg_rate * 0.1 * stress_ratio
        else:
            # Proportion above comfort: scales 0→∞ but we cap the egg output
            excess_ratio = (food_per_ant - comfort_food_per_ant) / comfort_food_per_ant
            expected_eggs = egg_rate * min(excess_ratio, 3.0)

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
