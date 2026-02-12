"""Tests for anthemyr.colony — Colony, Ant, Traits, Policies."""

from numpy.random import Generator

from anthemyr.colony.ant import Ant, Task
from anthemyr.colony.colony import Colony
from anthemyr.colony.policies import Policies
from anthemyr.colony.traits import Traits


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
        assert ant.task == Task.IDLE
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

    def test_remove_dead(self, default_colony: Colony) -> None:
        # Kill two ants
        default_colony.ants[0].hp = 0.0
        default_colony.ants[1].hp = 0.0
        dead = default_colony.remove_dead()
        assert len(dead) == 2
        assert all(not a.is_alive for a in dead)
        assert all(a.is_alive for a in default_colony.ants)
