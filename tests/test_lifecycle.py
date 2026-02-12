"""Tests for colony lifecycle - aging, starvation, food regen, brood."""

import pytest
from numpy.random import Generator

from anthemyr.colony.colony import Colony
from anthemyr.pheromones.fields import PheromoneType
from anthemyr.simulation.config import SimulationConfig
from anthemyr.simulation.engine import SimulationEngine
from anthemyr.world.world import World


class TestAging:
    """Tests for ant aging and natural death."""

    def test_aging_kills_old_ants(self, rng: Generator) -> None:
        """Ants that exceed max_age are killed."""
        colony = Colony(colony_id=0, nest_x=4, nest_y=4)
        ant = colony.spawn_ant(rng)
        ant.age = 500
        colony.apply_aging(max_age=500)
        assert ant.hp == 0.0
        assert not ant.is_alive

    def test_young_ants_survive(self, rng: Generator) -> None:
        """Ants below max_age are unaffected."""
        colony = Colony(colony_id=0, nest_x=4, nest_y=4)
        ant = colony.spawn_ant(rng)
        ant.age = 100
        colony.apply_aging(max_age=500)
        assert ant.hp > 0
        assert ant.is_alive

    def test_death_pheromone_deposited(self) -> None:
        """Engine deposits DEATH pheromone at corpse sites."""
        cfg = SimulationConfig(
            seed=42,
            world_width=8,
            world_height=8,
            initial_ants=3,
            max_age=5,
        )
        engine = SimulationEngine(config=cfg)
        colony = Colony(colony_id=0, nest_x=4, nest_y=4)
        engine.add_colony(colony)

        # Run enough ticks for ants to age out
        engine.run(ticks=10)

        # Check that DEATH pheromone was deposited somewhere
        death_grid = engine.pheromone_field.get_layer(PheromoneType.DEATH)
        assert death_grid.sum() > 0


class TestFoodPressure:
    """Tests for per-capita food pressure mechanics."""

    def test_pressure_damages_ants_when_food_low(self, rng: Generator) -> None:
        """Ants lose HP when food-per-ant is below comfort level."""
        colony = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            food_stores=1.0,  # 1.0 / 1 ant = 1.0, below comfort=2.0
        )
        ant = colony.spawn_ant(rng)
        initial_hp = ant.hp
        colony.apply_food_pressure(comfort_food_per_ant=2.0, max_damage=0.1)
        assert ant.hp < initial_hp

    def test_no_damage_when_food_comfortable(self, rng: Generator) -> None:
        """Ants are not damaged when food-per-ant exceeds comfort."""
        colony = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            food_stores=50.0,  # 50.0 / 1 ant = 50.0, well above comfort
        )
        ant = colony.spawn_ant(rng)
        initial_hp = ant.hp
        colony.apply_food_pressure(comfort_food_per_ant=2.0, max_damage=0.1)
        assert ant.hp == initial_hp

    def test_pressure_proportional_to_scarcity(self, rng: Generator) -> None:
        """More scarcity means more damage (smooth gradient)."""
        # Colony with half-comfort food (1.0 food / 1 ant = 1.0)
        colony_half = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            food_stores=1.0,
        )
        ant_half = colony_half.spawn_ant(rng)
        hp_before_half = ant_half.hp
        colony_half.apply_food_pressure(
            comfort_food_per_ant=2.0,
            max_damage=0.1,
        )
        damage_half = hp_before_half - ant_half.hp

        # Colony with zero food
        colony_zero = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            food_stores=0.0,
        )
        ant_zero = colony_zero.spawn_ant(rng)
        hp_before_zero = ant_zero.hp
        colony_zero.apply_food_pressure(
            comfort_food_per_ant=2.0,
            max_damage=0.1,
        )
        damage_zero = hp_before_zero - ant_zero.hp

        # Zero food should cause max damage
        assert damage_zero == pytest.approx(0.1)
        # Half-comfort should cause half the max damage
        assert damage_half == pytest.approx(0.05)
        # More scarcity → more damage
        assert damage_zero > damage_half

    def test_pressure_eventually_kills(self, rng: Generator) -> None:
        """Sustained pressure reduces HP to 0."""
        colony = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            food_stores=0.0,
        )
        ant = colony.spawn_ant(rng)
        for _ in range(20):
            colony.apply_food_pressure(
                comfort_food_per_ant=2.0,
                max_damage=0.1,
            )
        assert not ant.is_alive

    def test_engine_pressure_kills_ants(self) -> None:
        """Full engine integration: starving colony loses ants."""
        cfg = SimulationConfig(
            seed=42,
            world_width=8,
            world_height=8,
            initial_ants=10,
            max_age=99999,
            comfort_food_per_ant=2.0,
            max_starvation_damage=0.5,
            base_regen_rate=0.0,
            spread_regen_rate=0.0,
        )
        engine = SimulationEngine(config=cfg)
        colony = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            food_stores=0.0,
        )
        engine.add_colony(colony)
        engine.run(ticks=5)
        # With 0.5 max damage/tick and ~1.0 HP, ants die in 2 ticks
        assert len(colony.ants) < 10


class TestFoodRegeneration:
    """Tests for world food regeneration."""

    def test_food_spreads_from_existing(self, rng: Generator) -> None:
        """Food grows outward from cells that already have food."""
        world = World(width=8, height=8)
        # Seed a 3x3 cluster so the centre has full inner-ring density
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                world.cell_at(4 + dx, 4 + dy).food = 5.0
        total_before = sum(c.food for row in world.cells for c in row)
        for _ in range(100):
            world.regenerate_food(
                rng,
                base_rate=0.0,
                spread_rate=1.0,
                food_cap=5.0,
            )
        total_after = sum(c.food for row in world.cells for c in row)
        assert total_after > total_before
        # Edge neighbour of cluster should have food
        assert world.cell_at(2, 4).food > 0 or world.cell_at(6, 4).food > 0

    def test_no_spread_without_neighbours(self, rng: Generator) -> None:
        """Empty cells don't grow food without nearby food sources."""
        world = World(width=4, height=4)
        for _ in range(100):
            world.regenerate_food(
                rng,
                base_rate=0.0,
                spread_rate=0.5,
                food_cap=5.0,
            )
        total = sum(c.food for row in world.cells for c in row)
        assert total == 0.0

    def test_food_respects_cap(self, rng: Generator) -> None:
        """Food never exceeds the configured cap."""
        world = World(width=4, height=4)
        # Seed all cells so spread is active
        for row in world.cells:
            for cell in row:
                cell.food = 3.0
        for _ in range(1000):
            world.regenerate_food(
                rng,
                base_rate=0.0,
                spread_rate=1.0,
                food_cap=3.0,
            )
        for row in world.cells:
            for cell in row:
                assert cell.food <= 3.0

    def test_nest_cells_dont_regen(self, rng: Generator) -> None:
        """Nest cells should not accumulate food."""
        world = World(width=4, height=4)
        # Seed neighbours with food
        world.cell_at(1, 2).food = 5.0
        world.cell_at(3, 2).food = 5.0
        world.mark_nest(2, 2, radius=1)
        for _ in range(100):
            world.regenerate_food(
                rng,
                base_rate=1.0,
                spread_rate=1.0,
                food_cap=5.0,
            )
        nest_cell = world.cell_at(2, 2)
        assert nest_cell.food == 0.0

    def test_spontaneous_growth(self, rng: Generator) -> None:
        """Base rate allows rare spontaneous food on empty cells."""
        world = World(width=8, height=8)
        for _ in range(500):
            world.regenerate_food(
                rng,
                base_rate=1.0,
                spread_rate=0.0,
                food_cap=5.0,
            )
        total = sum(c.food for row in world.cells for c in row)
        assert total > 0


class TestBroodDevelopment:
    """Tests for queen egg-laying and brood maturation."""

    def test_no_eggs_when_severely_stressed(self, rng: Generator) -> None:
        """Colony doesn't lay eggs when food-per-ant is below 50% comfort."""
        colony = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            food_stores=2.0,
        )
        colony.spawn_ant(rng)  # 1 ant, food_per_ant = 2.0
        colony.spawn_ant(rng)  # 2 ants, food_per_ant = 1.0
        colony.spawn_ant(rng)  # 3 ants, food_per_ant = 0.67 < 1.0 (50% of 2.0)
        colony.lay_eggs(egg_rate=0.5, comfort_food_per_ant=2.0, rng=rng)
        assert colony.brood_count == 0

    def test_eggs_laid_with_surplus(self, rng: Generator) -> None:
        """Colony lays eggs when food-per-ant exceeds comfort."""
        colony = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            food_stores=60.0,
        )
        colony.spawn_ant(rng)  # 1 ant, food_per_ant = 60.0 >> 2.0
        colony.lay_eggs(egg_rate=0.5, comfort_food_per_ant=2.0, rng=rng)
        assert colony.brood_count > 0

    def test_egg_laying_costs_food(self, rng: Generator) -> None:
        """Each egg costs 1 food unit."""
        colony = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            food_stores=60.0,
        )
        colony.spawn_ant(rng)
        food_before = colony.food_stores
        colony.lay_eggs(egg_rate=0.5, comfort_food_per_ant=2.0, rng=rng)
        if colony.brood_count > 0:
            assert colony.food_stores < food_before

    def test_brood_matures_into_ants(self, rng: Generator) -> None:
        """Brood eventually hatches into new ants."""
        colony = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            brood_count=5,
        )
        initial_ants = len(colony.ants)
        # Mature ticks = 10; with 5 brood, progress += 5 each tick
        # After 2 ticks: progress = 10 -> hatch 1
        for _ in range(10):
            colony.develop_brood(mature_ticks=10, rng=rng)
        assert len(colony.ants) > initial_ants

    def test_brood_count_decreases_on_hatch(self, rng: Generator) -> None:
        """Brood count decreases as brood hatches."""
        colony = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            brood_count=3,
        )
        for _ in range(100):
            colony.develop_brood(mature_ticks=10, rng=rng)
        assert colony.brood_count == 0

    def test_no_brood_no_hatch(self, rng: Generator) -> None:
        """No brood means no new ants."""
        colony = Colony(
            colony_id=0,
            nest_x=4,
            nest_y=4,
            brood_count=0,
        )
        hatched = colony.develop_brood(mature_ticks=10, rng=rng)
        assert hatched == 0
        assert len(colony.ants) == 0

    def test_full_lifecycle_integration(self) -> None:
        """Integrated test: ants forage, colony breeds, population grows."""
        cfg = SimulationConfig(
            seed=42,
            world_width=16,
            world_height=16,
            initial_ants=10,
            max_age=99999,
            comfort_food_per_ant=2.0,
            max_starvation_damage=0.05,
            base_regen_rate=0.05,
            spread_regen_rate=0.1,
            food_cap=5.0,
            egg_rate=0.5,
            brood_mature_ticks=20,
        )
        engine = SimulationEngine(config=cfg)
        colony = Colony(
            colony_id=0,
            nest_x=8,
            nest_y=8,
            food_stores=200.0,
        )
        engine.add_colony(colony)
        engine.run(ticks=50)

        # With lots of food and high egg rate, colony should have grown
        assert len(colony.ants) > 10
        # Brood system was active
        assert colony.generation == 0  # no evolution yet


class TestLifecycleDeterminism:
    """Ensure lifecycle features preserve determinism."""

    def test_determinism_with_lifecycle(self) -> None:
        """Same seed produces identical state with birth/death."""
        cfg = SimulationConfig(
            seed=777,
            world_width=8,
            world_height=8,
            initial_ants=5,
            max_age=100,
            comfort_food_per_ant=2.0,
            max_starvation_damage=0.05,
            base_regen_rate=0.01,
            spread_regen_rate=0.05,
            food_cap=5.0,
            egg_rate=0.5,
            brood_mature_ticks=30,
        )

        engine_a = SimulationEngine(config=cfg)
        colony_a = Colony(colony_id=0, nest_x=4, nest_y=4)
        engine_a.add_colony(colony_a)
        engine_a.run(ticks=50)

        engine_b = SimulationEngine(config=cfg)
        colony_b = Colony(colony_id=0, nest_x=4, nest_y=4)
        engine_b.add_colony(colony_b)
        engine_b.run(ticks=50)

        assert len(colony_a.ants) == len(colony_b.ants)
        assert colony_a.food_stores == colony_b.food_stores
        assert colony_a.brood_count == colony_b.brood_count

        for ant_a, ant_b in zip(
            colony_a.ants,
            colony_b.ants,
            strict=True,
        ):
            assert ant_a.x == ant_b.x
            assert ant_a.y == ant_b.y
            assert ant_a.task == ant_b.task
            assert ant_a.hp == ant_b.hp
            assert ant_a.age == ant_b.age
