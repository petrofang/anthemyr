"""Tests for anthemyr.colony - Colony, Ant, Traits, Policies."""

from numpy.random import Generator

from anthemyr.colony.ant import Ant, Task
from anthemyr.colony.colony import Colony
from anthemyr.colony.policies import Policies
from anthemyr.colony.traits import Traits
from anthemyr.pheromones.fields import PheromoneField, PheromoneType
from anthemyr.world.world import World


class TestTraits:
    """Tests for the Traits dataclass."""

    def test_defaults_in_range(self) -> None:
        t = Traits()
        assert 0.0 <= t.aggression <= 1.0
        assert 0.0 <= t.exploration <= 1.0
        assert t.threshold_variance > 0


class TestPolicies:
    """Tests for the Policies dataclass."""

    def test_defaults_in_range(self) -> None:
        p = Policies()
        assert 0.0 <= p.aggression <= 1.0
        assert 0.0 <= p.caste_soldier_ratio <= 1.0


class TestAnt:
    """Tests for individual Ant agents."""

    def test_from_traits_sets_position(self, rng: Generator) -> None:
        ant = Ant.from_traits(x=5, y=3, traits=Traits(), rng=rng)
        assert ant.x == 5
        assert ant.y == 3

    def test_from_traits_has_thresholds(self, rng: Generator) -> None:
        ant = Ant.from_traits(x=0, y=0, traits=Traits(), rng=rng)
        assert "food" in ant.thresholds
        assert "alarm" in ant.thresholds
        assert "brood" in ant.thresholds
        assert "waste" in ant.thresholds

    def test_default_state(self, rng: Generator) -> None:
        ant = Ant.from_traits(x=0, y=0, traits=Traits(), rng=rng)
        assert ant.task == Task.FORAGING
        assert ant.is_alive
        assert ant.age == 0

    def test_dead_ant(self) -> None:
        ant = Ant(x=0, y=0, hp=0.0)
        assert not ant.is_alive


class TestColony:
    """Tests for Colony aggregate state."""

    def test_spawn_ant(self, rng: Generator) -> None:
        colony = Colony(colony_id=0, nest_x=4, nest_y=4)
        ant = colony.spawn_ant(rng)
        assert ant in colony.ants
        assert ant.x == 4
        assert ant.y == 4

    def test_consume_food(self, default_colony: Colony) -> None:
        initial = default_colony.food_stores
        default_colony.consume_food(amount_per_ant=1.0)
        expected = initial - len(default_colony.ants) * 1.0
        assert default_colony.food_stores == expected


class TestAntForaging:
    """Tests for ant foraging behaviour."""

    def test_idle_ant_can_start_foraging(self, rng: Generator) -> None:
        world = World(width=8, height=8)
        phero = PheromoneField(width=8, height=8)
        ant = Ant.from_traits(x=4, y=4, traits=Traits(), rng=rng)
        ant.task = Task.IDLE  # manually set IDLE to test transition
        ant.thresholds["food"] = 0.0  # zero threshold = always forages
        ant.update(world, phero, 4, 4, rng)
        assert ant.task == Task.FORAGING

    def test_foraging_ant_picks_up_food(self, rng: Generator) -> None:
        world = World(width=8, height=8)
        world.cell_at(3, 4).food = 5.0
        phero = PheromoneField(width=8, height=8)
        ant = Ant(x=3, y=4, task=Task.FORAGING)
        ant.update(world, phero, 4, 4, rng)
        assert ant.task == Task.CARRYING_FOOD
        assert ant.carrying_food > 0
        assert world.cell_at(3, 4).food < 5.0

    def test_carrying_ant_deposits_at_nest(self, rng: Generator) -> None:
        world = World(width=8, height=8)
        world.mark_nest(4, 4, radius=1)
        phero = PheromoneField(width=8, height=8)
        # Non-motherlode carrying ant (no _lay_trail) reverts to FORAGING
        ant = Ant(x=4, y=4, task=Task.CARRYING_FOOD, carrying_food=1.0)
        food = ant.update(world, phero, 4, 4, rng)
        assert ant.task == Task.FORAGING
        assert ant.carrying_food == 0.0
        assert food == 1.0

    def test_foraging_ant_deposits_trail_pheromone(
        self,
        rng: Generator,
    ) -> None:
        world = World(width=8, height=8)
        world.cell_at(3, 4).food = 5.0  # >= motherlode threshold (3.0)
        phero = PheromoneField(width=8, height=8)
        ant = Ant(x=3, y=4, task=Task.FORAGING)
        ant.update(world, phero, 4, 4, rng)
        # Ant picked up food from a motherlode and deposited trail pheromone
        assert phero.read(PheromoneType.TRAIL, 3, 4) > 0

    def test_small_food_no_trail(self, rng: Generator) -> None:
        """Small food sources don't deposit trail pheromone."""
        world = World(width=8, height=8)
        world.cell_at(3, 4).food = 1.5  # below motherlode threshold
        phero = PheromoneField(width=8, height=8)
        ant = Ant(x=3, y=4, task=Task.FORAGING)
        ant.update(world, phero, 4, 4, rng)
        assert ant.task == Task.CARRYING_FOOD
        assert phero.read(PheromoneType.TRAIL, 3, 4) == 0.0

    def test_ant_has_heading(self, rng: Generator) -> None:
        """Ants from_traits have a random heading."""
        ant = Ant.from_traits(x=0, y=0, traits=Traits(), rng=rng)
        assert 0.0 <= ant.heading < 2.0 * 3.15  # approx 2*pi

    def test_return_trip_reverses_heading(self, rng: Generator) -> None:
        """After depositing food, ant reverses heading."""
        import math

        world = World(width=8, height=8)
        world.mark_nest(4, 4, radius=1)
        phero = PheromoneField(width=8, height=8)
        ant = Ant(
            x=4,
            y=4,
            task=Task.CARRYING_FOOD,
            carrying_food=1.0,
            heading=0.0,
        )
        ant.update(world, phero, 4, 4, rng)
        # Heading should be reversed (0 -> pi)
        assert abs(ant.heading - math.pi) < 0.01

    def test_ant_ages_each_tick(self, rng: Generator) -> None:
        world = World(width=8, height=8)
        phero = PheromoneField(width=8, height=8)
        ant = Ant(x=4, y=4)
        ant.update(world, phero, 4, 4, rng)
        assert ant.age == 1

    def test_remove_dead(self, default_colony: Colony) -> None:
        # Kill two ants
        default_colony.ants[0].hp = 0.0
        default_colony.ants[1].hp = 0.0
        dead = default_colony.remove_dead()
        assert len(dead) == 2
        assert all(not a.is_alive for a in dead)
        assert all(a.is_alive for a in default_colony.ants)


class TestGathering:
    """Tests for the GATHERING task mode."""

    def test_motherlode_forager_becomes_gatherer(self, rng: Generator) -> None:
        """After depositing motherlode food at nest, ant becomes GATHERING."""
        world = World(width=8, height=8)
        world.mark_nest(4, 4, radius=1)
        phero = PheromoneField(width=8, height=8)
        ant = Ant(
            x=4,
            y=4,
            task=Task.CARRYING_FOOD,
            carrying_food=3.0,
            heading=0.0,
            _lay_trail=True,
        )
        ant.update(world, phero, 4, 4, rng)
        assert ant.task == Task.GATHERING
        assert ant.carrying_food == 0.0

    def test_small_find_forager_stays_foraging(self, rng: Generator) -> None:
        """After depositing non-motherlode food, ant reverts to FORAGING."""
        world = World(width=8, height=8)
        world.mark_nest(4, 4, radius=1)
        phero = PheromoneField(width=8, height=8)
        ant = Ant(
            x=4,
            y=4,
            task=Task.CARRYING_FOOD,
            carrying_food=1.0,
            heading=0.0,
            _lay_trail=False,
        )
        ant.update(world, phero, 4, 4, rng)
        assert ant.task == Task.FORAGING

    def test_gatherer_picks_up_food(self, rng: Generator) -> None:
        """GATHERING ant picks up food and switches to CARRYING_FOOD."""
        world = World(width=8, height=8)
        world.cell_at(3, 3).food = 5.0
        phero = PheromoneField(width=8, height=8)
        ant = Ant(x=3, y=3, task=Task.GATHERING, _lay_trail=True)
        ant.update(world, phero, 4, 4, rng)
        assert ant.task == Task.CARRYING_FOOD
        assert ant.carrying_food > 0

    def test_gatherer_patience_expires(self, rng: Generator) -> None:
        """GATHERING ant with zero patience reverts to FORAGING in place."""
        world = World(width=8, height=8)
        phero = PheromoneField(width=8, height=8)
        ant = Ant(
            x=2,
            y=2,
            task=Task.GATHERING,
            _gather_patience=1,
            _lay_trail=True,
        )
        ant.update(world, phero, 4, 4, rng)
        assert ant.task == Task.FORAGING
        assert not ant._lay_trail

    def test_gatherer_reinforces_trail(self, rng: Generator) -> None:
        """GATHERING ant deposits trail pheromone at lower rate while walking."""
        world = World(width=8, height=8)
        phero = PheromoneField(width=8, height=8)
        ant = Ant(
            x=3,
            y=3,
            task=Task.GATHERING,
            _lay_trail=True,
            _gather_patience=60,
        )
        ant.update(world, phero, 4, 4, rng)
        # Ant is still gathering and moved -- check trail deposit at new pos
        assert ant.task == Task.GATHERING
        trail = phero.read(PheromoneType.TRAIL, ant.x, ant.y)
        assert trail > 0  # trail deposited
        assert trail <= 1.0  # at the lower gatherer rate (1.0), not scout (2.0)

    def test_recruitment_pheromone_recruits_idle_ant(
        self,
        rng: Generator,
    ) -> None:
        """Strong RECRUITMENT pheromone can switch an IDLE ant to GATHERING."""
        world = World(width=8, height=8)
        phero = PheromoneField(width=8, height=8)
        # Deposit strong recruitment signal
        phero.deposit(PheromoneType.RECRUITMENT, 4, 4, 20.0)
        ant = Ant(x=4, y=4, task=Task.IDLE)
        ant.thresholds["food"] = 0.01  # very eager ant
        # Run several ticks to account for probabilistic switching
        switched = False
        for _ in range(20):
            ant.task = Task.IDLE
            ant.update(world, phero, 4, 4, rng)
            if ant.task == Task.GATHERING:
                switched = True
                break
        assert switched, "Idle ant should eventually respond to recruitment"

    def test_recruitment_pheromone_recruits_foraging_ant(
        self,
        rng: Generator,
    ) -> None:
        """FORAGING ant near recruitment pheromone can switch to GATHERING."""
        world = World(width=8, height=8)
        phero = PheromoneField(width=8, height=8)
        phero.deposit(PheromoneType.RECRUITMENT, 3, 3, 20.0)
        ant = Ant(x=3, y=3, task=Task.FORAGING)
        ant.thresholds["food"] = 0.01
        ant._forage_ticks = 50  # long unsuccessful search = more susceptible
        switched = False
        for _ in range(20):
            ant.task = Task.FORAGING
            ant.x, ant.y = 3, 3
            ant._forage_ticks = 50
            ant.update(world, phero, 4, 4, rng)
            if ant.task == Task.GATHERING:
                switched = True
                break
        assert switched, "Foraging ant should respond to recruitment"
