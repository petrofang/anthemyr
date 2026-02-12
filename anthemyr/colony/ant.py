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
    from anthemyr.pheromones.fields import PheromoneField
    from anthemyr.world.cell import Cell
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
            "alarm": float(rng.normal(traits.alarm_threshold_mean, var)),
            "brood": float(rng.normal(traits.brood_care_threshold_mean, var)),
            "waste": float(rng.normal(traits.waste_threshold_mean, var)),
        }
        hp = float(rng.uniform(0.8, 1.2))
        return cls(x=x, y=y, hp=hp, thresholds=thresholds)

    def update(
        self,
        world: World,
        pheromones: PheromoneField,
        nest_x: int,
        nest_y: int,
        rng: Generator,
    ) -> float:
        """Perform one tick of local decision-making and movement.

        Args:
            world: The world grid for spatial queries.
            pheromones: Multi-layer pheromone field for reading/depositing.
            nest_x: Colony nest X coordinate (for homing).
            nest_y: Colony nest Y coordinate (for homing).
            rng: Seeded random generator.

        Returns:
            Amount of food deposited at the nest (0.0 if not returning).
        """
        from anthemyr.pheromones.fields import PheromoneType

        self.age += 1
        food_deposited = 0.0

        match self.task:
            case Task.IDLE:
                self._decide_to_forage(world, pheromones, rng)
            case Task.FORAGING:
                self._forage(world, pheromones, rng)
            case Task.CARRYING_FOOD:
                food_deposited = self._carry_food_home(
                    world,
                    pheromones,
                    nest_x,
                    nest_y,
                    rng,
                )

        # Trail pheromone deposit while carrying food (breadcrumb home)
        if self.task == Task.CARRYING_FOOD:
            pheromones.deposit(
                PheromoneType.TRAIL,
                self.x,
                self.y,
                1.0,
            )

        return food_deposited

    # -- Private behaviour methods --

    def _decide_to_forage(
        self,
        world: World,
        pheromones: PheromoneField,
        rng: Generator,
    ) -> None:
        """IDLE ants evaluate whether to start foraging.

        An ant starts foraging when a random stimulus draw exceeds
        its food-response threshold (lower threshold = more eager).
        """
        stimulus = float(rng.random())
        threshold = self.thresholds.get("food", 0.5)
        if stimulus > threshold:
            self.task = Task.FORAGING

    def _forage(
        self,
        world: World,
        pheromones: PheromoneField,
        rng: Generator,
    ) -> None:
        """FORAGING ants walk, biased toward trail pheromone, seeking food."""
        from anthemyr.pheromones.fields import PheromoneType

        cell = world.cell_at(self.x, self.y)

        # Pick up food if present
        if cell.food > 0 and not cell.is_nest:
            pickup = min(cell.food, 1.0)
            cell.food -= pickup
            self.carrying_food = pickup
            self.task = Task.CARRYING_FOOD
            # Strong trail deposit at food source
            pheromones.deposit(
                PheromoneType.TRAIL,
                self.x,
                self.y,
                3.0,
            )
            return

        # Move: bias toward trail pheromone, otherwise random walk
        self._move_biased(world, pheromones, PheromoneType.TRAIL, rng)

    def _carry_food_home(
        self,
        world: World,
        pheromones: PheromoneField,
        nest_x: int,
        nest_y: int,
        rng: Generator,
    ) -> float:
        """CARRYING_FOOD ants walk toward the nest and deposit food.

        Returns:
            Amount of food deposited (> 0 only when reaching nest).
        """
        # Arrived at nest?
        if world.cell_at(self.x, self.y).is_nest:
            deposited = self.carrying_food
            self.carrying_food = 0.0
            self.task = Task.IDLE
            return deposited

        # Move toward nest with some randomness
        self._move_toward(world, nest_x, nest_y, rng)
        return 0.0

    def _move_biased(
        self,
        world: World,
        pheromones: PheromoneField,
        bias_type: object,
        rng: Generator,
    ) -> None:
        """Move to a neighbour, biased by pheromone concentration.

        With probability 0.3, follows the strongest pheromone gradient;
        otherwise picks a random neighbour.  This balance between
        exploitation and exploration is key to emergent trail formation.
        """
        from anthemyr.pheromones.fields import PheromoneType

        neighbours = world.neighbours(self.x, self.y)
        if not neighbours:
            return

        if rng.random() < 0.3 and isinstance(bias_type, PheromoneType):
            best = self._best_pheromone_neighbour(
                neighbours,
                pheromones,
                bias_type,
            )
            if best is not None:
                self.x, self.y = best.x, best.y
                return

        # Random walk
        chosen = neighbours[int(rng.integers(len(neighbours)))]
        self.x, self.y = chosen.x, chosen.y

    def _move_toward(
        self,
        world: World,
        target_x: int,
        target_y: int,
        rng: Generator,
    ) -> None:
        """Move one step toward a target with slight randomness."""
        neighbours = world.neighbours(self.x, self.y)
        if not neighbours:
            return

        # Sort by Manhattan distance to target, pick from best few
        neighbours.sort(
            key=lambda c: abs(c.x - target_x) + abs(c.y - target_y),
        )
        # Pick from the closest 1-3 neighbours for some variance
        top = min(3, len(neighbours))
        chosen = neighbours[int(rng.integers(top))]
        self.x, self.y = chosen.x, chosen.y

    @staticmethod
    def _best_pheromone_neighbour(
        neighbours: list[Cell],
        pheromones: PheromoneField,
        ptype: object,
    ) -> Cell | None:
        """Return the neighbour with the highest pheromone of the given type.

        Returns None if all neighbours have zero concentration.
        """
        from anthemyr.pheromones.fields import PheromoneType

        if not isinstance(ptype, PheromoneType):
            return None

        best_cell: Cell | None = None
        best_val = 0.0
        for cell in neighbours:
            val = pheromones.read(ptype, cell.x, cell.y)
            if val > best_val:
                best_val = val
                best_cell = cell
        return best_cell
