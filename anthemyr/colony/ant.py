"""Ant -- individual agent with local decision-making.

Each Ant carries internal response thresholds that determine when it
switches between tasks.  Behaviour emerges from the interplay of these
thresholds with the local pheromone landscape and environmental stimuli.

Key movement model:

- **Correlated random walk**: each ant maintains a *heading* (radians)
  and biases movement toward it with Gaussian angular noise.  This
  produces directional persistence and much better area coverage than
  an uncorrelated random walk.
- **Explored pheromone**: foraging ants deposit a light TERRITORY mark
  each step.  Movement is repelled by TERRITORY so scouts naturally
  spread out and avoid re-searching the same ground.
- **Return-trip memory**: after delivering food to the nest the ant
  reverses its heading to walk back toward the food source it just
  exploited, rather than starting a fresh random search.
- **Gathering mode**: successful foragers that find a motherlode
  transition to GATHERING after depositing food at the nest.  Gatherers
  follow TRAIL/RECRUITMENT pheromone at 75% and reinforce the trail.
  Other ants (scouts and idle) can be recruited into GATHERING by
  strong RECRUITMENT pheromone, modulated by personal thresholds and
  how long they have been searching unsuccessfully.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

    from anthemyr.colony.traits import Traits
    from anthemyr.pheromones.fields import PheromoneField
    from anthemyr.world.cell import Cell
    from anthemyr.world.world import World

# -- Constants ---------------------------------------------------------------

_FORAGE_FOLLOW_RATE = 0.40  # trail-follow probability while searching
_CARRY_FOLLOW_RATE = 0.75  # trail-follow probability while carrying food
_GATHER_FOLLOW_RATE = 0.75  # trail/recruitment follow while gathering
_GATHER_TRAIL_DEPOSIT = 1.0  # trail reinforcement per step while gathering
_GATHER_PATIENCE_MAX = 60  # ticks a gatherer walks without food before giving up
_RECRUITMENT_SWITCH_BASE = 0.3  # base recruitment-response probability
_HEADING_NOISE_STD = 0.5  # ~30 degrees Gaussian noise on heading
_TERRITORY_DEPOSIT = 0.3  # explored-area mark per step
_MOTHERLODE_THRESHOLD = 3.0  # food remaining to count as "motherlode"
_TRAIL_DEPOSIT_AT_FOOD = 5.0
_RECRUITMENT_DEPOSIT = 3.0
_TRAIL_DEPOSIT_PER_STEP = 2.0
_FOOD_PICKUP = 3.0


class Task(Enum):
    """Behavioural task an ant is currently performing."""

    IDLE = auto()
    FORAGING = auto()
    GATHERING = auto()
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
        heading: Current movement direction in radians (0 = east,
            pi/2 = south).  Used for correlated random walk.
        thresholds: Per-stimulus response thresholds.  Lower values mean
            the ant is more likely to respond to that stimulus.
        _lay_trail: Whether to deposit trail while carrying/gathering.
        _gather_patience: Ticks remaining before a gatherer gives up
            and reverts to scouting from its current position.
        _forage_ticks: Consecutive ticks spent foraging without finding
            food.  Higher values make the ant more susceptible to
            recruitment pheromone.
    """

    x: int
    y: int
    task: Task = Task.IDLE
    hp: float = 1.0
    age: int = 0
    carrying_food: float = 0.0
    heading: float = 0.0
    _lay_trail: bool = False
    _gather_patience: int = _GATHER_PATIENCE_MAX
    _forage_ticks: int = 0
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

        New ants start in FORAGING state with a random heading so they
        immediately scatter outward in different directions.

        Args:
            x: Spawn column.
            y: Spawn row.
            traits: Colony-wide genetic trait profile.
            rng: Seeded random generator.

        Returns:
            A new Ant instance with randomised thresholds and heading.
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
        heading = float(rng.uniform(0.0, 2.0 * math.pi))
        return cls(
            x=x,
            y=y,
            task=Task.FORAGING,
            hp=hp,
            heading=heading,
            thresholds=thresholds,
        )

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
            case Task.GATHERING:
                self._gather(world, pheromones, nest_x, nest_y, rng)
            case Task.CARRYING_FOOD:
                food_deposited = self._carry_food_home(
                    world,
                    pheromones,
                    nest_x,
                    nest_y,
                    rng,
                )

        # Trail pheromone deposit while carrying food (breadcrumb home).
        # Only lay trail when the food source was a motherlode.
        if self.task == Task.CARRYING_FOOD and self._lay_trail:
            pheromones.deposit(
                PheromoneType.TRAIL,
                self.x,
                self.y,
                _TRAIL_DEPOSIT_PER_STEP,
            )

        # Gatherers reinforce the trail at a lower rate so successful
        # routes stay alive while the food source remains productive.
        if self.task == Task.GATHERING and self._lay_trail:
            pheromones.deposit(
                PheromoneType.TRAIL,
                self.x,
                self.y,
                _GATHER_TRAIL_DEPOSIT,
            )

        return food_deposited

    # -- Private behaviour methods --

    def _decide_to_forage(
        self,
        world: World,
        pheromones: PheromoneField,
        rng: Generator,
    ) -> None:
        """IDLE ants evaluate whether to start foraging or gathering.

        An ant starts foraging when a random stimulus draw exceeds
        its food-response threshold (lower threshold = more eager).
        If strong RECRUITMENT pheromone is present at the ant's cell,
        it may skip straight to GATHERING instead.
        """
        from anthemyr.pheromones.fields import PheromoneType

        # Check for recruitment pheromone first -- strong signal can
        # pull an idle ant directly into gathering mode.
        recruitment = pheromones.read(
            PheromoneType.RECRUITMENT,
            self.x,
            self.y,
        )
        if recruitment > 0:
            threshold = self.thresholds.get("food", 0.5)
            # Sigmoid-like response: strong signal + low threshold = likely
            prob = recruitment / (recruitment + threshold)
            if rng.random() < prob:
                self.task = Task.GATHERING
                self._lay_trail = True
                self._gather_patience = _GATHER_PATIENCE_MAX
                return

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
        """FORAGING ants walk, seeking food with correlated random walk.

        Movement strategy:

        1. If standing on food, pick it up.  Only deposit a heavy trail
           + RECRUITMENT for a "motherlode" (>=3.0 food remaining).
           Small finds are picked up silently so they don't create
           misleading trails to already-depleted sources.
        2. Check for RECRUITMENT pheromone -- a scout that has been
           searching unsuccessfully for a long time becomes increasingly
           susceptible to switching to GATHERING mode.
        3. Otherwise move using ``_move_foraging`` -- a correlated
           random walk biased slightly toward trail pheromone (40%)
           and repelled by TERRITORY (explored-area) pheromone.
        4. Deposit a light TERRITORY mark so other scouts avoid
           re-searching this cell.
        """
        from anthemyr.pheromones.fields import PheromoneType

        cell = world.cell_at(self.x, self.y)

        # Pick up food if present
        if cell.food > 0 and not cell.is_nest:
            remaining_before = cell.food
            pickup = min(cell.food, _FOOD_PICKUP)
            cell.food -= pickup
            self.carrying_food = pickup
            self.task = Task.CARRYING_FOOD
            self._forage_ticks = 0  # reset: we found food

            # Motherlode: rich source gets a heavy trail + recruitment
            if remaining_before >= _MOTHERLODE_THRESHOLD:
                self._lay_trail = True
                pheromones.deposit(
                    PheromoneType.TRAIL,
                    self.x,
                    self.y,
                    _TRAIL_DEPOSIT_AT_FOOD,
                )
                pheromones.deposit(
                    PheromoneType.RECRUITMENT,
                    self.x,
                    self.y,
                    _RECRUITMENT_DEPOSIT,
                )
            # Small finds: no trail deposit -- silent pickup
            return

        # Track unsuccessful foraging time
        self._forage_ticks += 1

        # Check recruitment pheromone -- fuzzy switch to GATHERING.
        # Probability increases with:
        #   - stronger recruitment signal
        #   - longer time spent searching unsuccessfully (_forage_ticks)
        #   - lower personal food threshold (more eager ant)
        recruitment = pheromones.read(
            PheromoneType.RECRUITMENT,
            self.x,
            self.y,
        )
        if recruitment > 0:
            threshold = self.thresholds.get("food", 0.5)
            # Urgency ramps up: after 30 ticks of fruitless search the
            # ant is twice as susceptible, after 60 ticks three times.
            urgency = 1.0 + self._forage_ticks / 30.0
            prob = (
                _RECRUITMENT_SWITCH_BASE
                * urgency
                * recruitment
                / (recruitment + threshold)
            )
            if rng.random() < prob:
                self.task = Task.GATHERING
                self._lay_trail = True
                self._gather_patience = _GATHER_PATIENCE_MAX
                self._forage_ticks = 0
                return

        # Deposit explored-area mark before moving
        pheromones.deposit(
            PheromoneType.TERRITORY,
            self.x,
            self.y,
            _TERRITORY_DEPOSIT,
        )

        # Move: correlated walk with trail bias and territory avoidance
        self._move_foraging(world, pheromones, rng)

    def _carry_food_home(
        self,
        world: World,
        pheromones: PheromoneField,
        nest_x: int,
        nest_y: int,
        rng: Generator,
    ) -> float:
        """CARRYING_FOOD ants walk toward the nest and deposit food.

        After depositing, the ant transitions based on how it found
        the food:

        - **Motherlode** (``_lay_trail`` is True): switch to GATHERING
          and reverse heading to walk back toward the food source,
          exploiting the trail it just laid.
        - **Small find**: revert to FORAGING and start a fresh scout
          walk.

        Returns:
            Amount of food deposited (> 0 only when reaching nest).
        """
        # Arrived at nest?
        if world.cell_at(self.x, self.y).is_nest:
            deposited = self.carrying_food
            self.carrying_food = 0.0
            # Reverse heading to walk back toward the food source
            self.heading = (self.heading + math.pi) % (2.0 * math.pi)

            if self._lay_trail:
                # Successful motherlode trip -- become a gatherer
                self.task = Task.GATHERING
                self._gather_patience = _GATHER_PATIENCE_MAX
                # Keep _lay_trail True so we reinforce trail on the way
            else:
                # Small find -- go back to scouting
                self.task = Task.FORAGING
            return deposited

        # Move toward nest with some randomness
        self._move_toward(world, nest_x, nest_y, rng)
        return 0.0

    def _gather(
        self,
        world: World,
        pheromones: PheromoneField,
        nest_x: int,
        nest_y: int,
        rng: Generator,
    ) -> None:
        """GATHERING ants exploit a known food source via pheromone trails.

        Unlike foraging scouts, gatherers follow TRAIL and RECRUITMENT
        pheromone at a high rate (75%) to efficiently reach a food
        source discovered by a scout.  They reinforce the trail as
        they walk (handled in ``update()``).  Movement is directional
        -- gatherers prefer trail cells that lead *away from* the nest,
        avoiding the trap of oscillating on trail fragments near the
        nest entrance.

        **Food selection**: Gatherers only pick up food when it's
        worthwhile — either a motherlode (≥3.0) or part of a dense
        food cluster.  This prevents distraction by isolated crumbs
        along the way.  Dense cluster = 3+ adjacent cells with food.

        If the gatherer finds food it picks it up, resets its patience
        counter, and switches to CARRYING_FOOD.  If it walks for
        ``_GATHER_PATIENCE_MAX`` ticks without finding food (the source
        was depleted or the trail evaporated), it gives up and reverts
        to FORAGING from its *current position* -- becoming a scout
        that starts searching from where it expected the food to be,
        rather than retreating to the nest.
        """
        from anthemyr.pheromones.fields import PheromoneType

        cell = world.cell_at(self.x, self.y)

        # Check if worth picking up: motherlode OR part of dense cluster
        if cell.food > 0 and not cell.is_nest:
            remaining_before = cell.food
            is_motherlode = remaining_before >= _MOTHERLODE_THRESHOLD
            is_dense_cluster = self._is_in_food_cluster(world, self.x, self.y)

            if is_motherlode or is_dense_cluster:
                # Worth picking up -- commit to carrying
                pickup = min(cell.food, _FOOD_PICKUP)
                cell.food -= pickup
                self.carrying_food = pickup
                self.task = Task.CARRYING_FOOD
                self._gather_patience = _GATHER_PATIENCE_MAX

                # Refresh recruitment signal if still a motherlode
                if is_motherlode:
                    self._lay_trail = True
                    pheromones.deposit(
                        PheromoneType.TRAIL,
                        self.x,
                        self.y,
                        _TRAIL_DEPOSIT_AT_FOOD,
                    )
                    pheromones.deposit(
                        PheromoneType.RECRUITMENT,
                        self.x,
                        self.y,
                        _RECRUITMENT_DEPOSIT,
                    )
                else:
                    # Dense cluster, not motherlode -- still lay trail
                    # but more softly to avoid confusing scouts
                    self._lay_trail = True
                return
            # Small food on isolated cell -- ignore, stay on trail

        # No food here -- count down patience
        self._gather_patience -= 1
        if self._gather_patience <= 0:
            # Give up: revert to scouting from current position
            self.task = Task.FORAGING
            self._lay_trail = False
            self._forage_ticks = 0
            return

        # Move toward food source via pheromone trail
        self._move_gathering(world, pheromones, nest_x, nest_y, rng)

    def _move_gathering(
        self,
        world: World,
        pheromones: PheromoneField,
        nest_x: int,
        nest_y: int,
        rng: Generator,
    ) -> None:
        """Move as a gatherer: follow trail *away from* the nest.

        Gatherers use directional trail-following -- they prefer trail
        cells that lead away from the nest, avoiding the trap of
        oscillating on strong trail fragments near the nest entrance.
        This mirrors real ants using polarity cues (sun compass +
        trail gradient) to determine which direction to walk along
        a bidirectional pheromone trail.

        Decision order:

        1. With probability ``_GATHER_FOLLOW_RATE`` (75%), follow the
           best directional trail (TRAIL, then RECRUITMENT fallback).
           "Best" combines pheromone strength with preference for cells
           farther from the nest.
        2. Otherwise, fall back to a correlated random walk.
        """
        neighbours = world.neighbours(self.x, self.y)
        if not neighbours:
            return

        # 1. Directional pheromone following (75%) -- prefer away from nest
        if rng.random() < _GATHER_FOLLOW_RATE:
            best = self._best_directional_trail(
                neighbours,
                pheromones,
                nest_x,
                nest_y,
                toward_nest=False,
            )
            if best is not None:
                self._step_to(best)
                return

        # 2. Correlated walk fallback
        self._correlated_step(world, pheromones, rng)

    def _move_foraging(
        self,
        world: World,
        pheromones: PheromoneField,
        rng: Generator,
    ) -> None:
        """Move using a correlated random walk with pheromone cues.

        Decision order:

        1. With probability ``_FORAGE_FOLLOW_RATE`` (40%), follow the
           strongest TRAIL pheromone gradient (if any).  This is a
           moderate exploitation rate -- enough to follow real trails
           but not so high that ants cluster on noise.
        2. Otherwise, use a **correlated random walk**: pick the
           neighbour closest to the current heading (with Gaussian
           angular noise), preferring cells with *lower* TERRITORY
           concentration so scouts spread out.

        After moving, update ``heading`` to match the step taken.
        """
        from anthemyr.pheromones.fields import PheromoneType

        neighbours = world.neighbours(self.x, self.y)
        if not neighbours:
            return

        # 1. Trail following (exploitation) -- 40% chance
        if rng.random() < _FORAGE_FOLLOW_RATE:
            best = self._best_pheromone_neighbour(
                neighbours,
                pheromones,
                PheromoneType.TRAIL,
            )
            if best is not None:
                self._step_to(best)
                return

        # 2. Correlated random walk with territory avoidance
        self._correlated_step(world, pheromones, rng)

    def _correlated_step(
        self,
        world: World,
        pheromones: PheromoneField,
        rng: Generator,
    ) -> None:
        """Take one step biased toward current heading, avoiding explored area.

        For each neighbour, compute a score combining:
        - **Heading alignment**: cos(angle_to_neighbour - noisy_heading).
          Higher = better aligned with where the ant wants to go.
        - **Territory penalty**: subtract TERRITORY concentration so
          already-explored cells are less attractive.

        Pick the neighbour with the best combined score.
        """
        from anthemyr.pheromones.fields import PheromoneType

        neighbours = world.neighbours(self.x, self.y)
        if not neighbours:
            return

        # Noisy heading: current heading + Gaussian noise
        noisy = self.heading + float(rng.normal(0.0, _HEADING_NOISE_STD))

        best_cell: Cell | None = None
        best_score = -999.0
        for cell in neighbours:
            dx = cell.x - self.x
            dy = cell.y - self.y
            angle = math.atan2(dy, dx)
            # Heading alignment: 1.0 when perfectly aligned, -1.0 opposite
            alignment = math.cos(angle - noisy)
            # Territory penalty: prefer unexplored cells
            territory = pheromones.read(PheromoneType.TERRITORY, cell.x, cell.y)
            score = alignment - territory
            if score > best_score:
                best_score = score
                best_cell = cell

        if best_cell is not None:
            self._step_to(best_cell)

    def _step_to(self, cell: Cell) -> None:
        """Move to a cell and update heading to match the step direction."""
        dx = cell.x - self.x
        dy = cell.y - self.y
        if dx != 0 or dy != 0:
            self.heading = math.atan2(dy, dx)
        self.x, self.y = cell.x, cell.y

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
        # Pick from the closest 1-2 neighbours for faster returns
        top = min(2, len(neighbours))
        chosen = neighbours[int(rng.integers(top))]
        self._step_to(chosen)

    def _is_in_food_cluster(self, world: World, x: int, y: int) -> bool:
        """Check if a cell is part of a dense food cluster.

        Dense cluster = 3+ adjacent cells (including current cell) have
        food > 0.  This helps gatherers distinguish between isolated
        food crumbs and real food sources.

        Args:
            world: The world grid.
            x: Cell X coordinate.
            y: Cell Y coordinate.

        Returns:
            True if this cell is part of a cluster, False otherwise.
        """
        neighbours = world.neighbours(x, y)
        # Count cells with food, including current cell
        food_count = 1 if world.cell_at(x, y).food > 0 else 0
        for cell in neighbours:
            if cell.food > 0:
                food_count += 1
        return food_count >= 3

    def _best_directional_trail(
        self,
        neighbours: list[Cell],
        pheromones: PheromoneField,
        nest_x: int,
        nest_y: int,
        *,
        toward_nest: bool,
    ) -> Cell | None:
        """Pick the best trail-following neighbour with directional bias.

        Real ants can determine trail polarity — they don't just walk
        toward the strongest pheromone; they walk along the trail in
        the right direction.  This method scores each neighbour by
        combining trail pheromone strength with a directional bonus
        that prefers movement toward or away from the nest.

        The method checks TRAIL first; if no TRAIL is found on any
        neighbour it falls back to RECRUITMENT pheromone.

        Args:
            neighbours: Adjacent cells to choose from.
            pheromones: Multi-layer pheromone field.
            nest_x: Colony nest X coordinate.
            nest_y: Colony nest Y coordinate.
            toward_nest: If True, prefer cells closer to the nest
                (for carrying food home).  If False, prefer cells
                farther from the nest (for gathering outbound).

        Returns:
            The best neighbouring cell, or None if no pheromone found.
        """
        from anthemyr.pheromones.fields import PheromoneType

        # Current distance from nest
        my_dist = abs(self.x - nest_x) + abs(self.y - nest_y)

        for ptype in (PheromoneType.TRAIL, PheromoneType.RECRUITMENT):
            best_cell: Cell | None = None
            best_score = -999.0
            has_pheromone = False

            for cell in neighbours:
                val = pheromones.read(ptype, cell.x, cell.y)
                if val <= 0:
                    continue
                has_pheromone = True

                # Directional bonus: +1 if moving the right way, -1 if wrong
                cell_dist = abs(cell.x - nest_x) + abs(cell.y - nest_y)
                if toward_nest:
                    direction = 1.0 if cell_dist < my_dist else -0.5
                else:
                    direction = 1.0 if cell_dist > my_dist else -0.5

                # Combine: pheromone strength (normalised) + direction bias
                score = val + direction * 2.0
                if score > best_score:
                    best_score = score
                    best_cell = cell

            if has_pheromone:
                return best_cell

        return None

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
