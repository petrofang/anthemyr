"""Ant — individual agent with local decision-making.

Each Ant carries internal response thresholds that determine when it
switches between tasks.  Behaviour emerges from the interplay of these
thresholds with the local pheromone landscape and environmental stimuli.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

    from anthemyr.colony.traits import Traits
    from anthemyr.world.world import World


class Task(Enum):
    """Behavioural task an ant is currently performing."""

    IDLE = auto()
    FORAGING = auto()
    CARRYING_FOOD = auto()
    BROOD_CARE = auto()
    PATROLLING = auto()
    FIGHTING = auto()
    WASTE_MANAGEMENT = auto()


@dataclass
class Ant:
    """A single ant agent.

    Attributes:
        x: Current column position in the world grid.
        y: Current row position in the world grid.
        task: Current behavioural task.
        hp: Hit-points / vitality (dies at 0).
        age: Ticks since birth.
        carrying_food: Amount of food currently carried.
        thresholds: Per-stimulus response thresholds.  Lower values mean
            the ant is more likely to respond to that stimulus.
    """

    x: int
    y: int
    task: Task = Task.IDLE
    hp: float = 1.0
    age: int = 0
    carrying_food: float = 0.0
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        """Return True if this ant is still alive."""
        return self.hp > 0

    @classmethod
    def from_traits(
        cls,
        x: int,
        y: int,
        traits: Traits,
        rng: Generator,
    ) -> Ant:
        """Create an ant with thresholds sampled from colony trait distributions.

        Args:
            x: Spawn column.
            y: Spawn row.
            traits: Colony-wide genetic trait profile.
            rng: Seeded random generator.

        Returns:
            A new Ant instance with randomised thresholds.
        """
        mean = traits.foraging_threshold_mean
        var = traits.threshold_variance
        thresholds = {
            "food": float(rng.normal(mean, var)),
            "alarm": float(
                rng.normal(traits.alarm_threshold_mean, var)
            ),
            "brood": float(
                rng.normal(traits.brood_care_threshold_mean, var)
            ),
            "waste": float(
                rng.normal(traits.waste_threshold_mean, var)
            ),
        }
        return cls(x=x, y=y, thresholds=thresholds)

    def update(self, world: World, rng: Generator) -> None:
        """Perform one tick of local decision-making and movement.

        Args:
            world: The world grid for spatial queries.
            rng: Seeded random generator.
        """
        self.age += 1
        # TODO: Implement stimulus evaluation, task switching, and movement
