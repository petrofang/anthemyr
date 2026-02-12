"""Tests for anthemyr.simulation — engine and config loading."""

from pathlib import Path

import numpy as np

from anthemyr.colony.colony import Colony
from anthemyr.simulation.config import SimulationConfig
from anthemyr.simulation.engine import SimulationEngine


class TestSimulationConfig:
    """Tests for YAML config loading."""

    def test_defaults(self) -> None:
        cfg = SimulationConfig()
        assert cfg.seed == 42
        assert cfg.world_width == 64
        assert cfg.world_height == 64

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("seed: 99\nworld_width: 16\nworld_height: 16\n")
        cfg = SimulationConfig.from_yaml(yaml_file)
        assert cfg.seed == 99
        assert cfg.world_width == 16


class TestSimulationEngine:
    """Tests for the tick loop."""

    def test_engine_initialises(self, default_config: SimulationConfig) -> None:
        engine = SimulationEngine(config=default_config)
        assert engine.tick == 0
        assert engine.world.width == default_config.world_width

    def test_step_advances_tick(self, default_config: SimulationConfig) -> None:
        engine = SimulationEngine(config=default_config)
        engine.step()
        assert engine.tick == 1

    def test_run_multiple_ticks(self, default_config: SimulationConfig) -> None:
        engine = SimulationEngine(config=default_config)
        engine.run(ticks=10)
        assert engine.tick == 10

    def test_determinism(self) -> None:
        """Same seed must produce identical state after N ticks."""
        cfg = SimulationConfig(
            seed=777,
            world_width=8,
            world_height=8,
            initial_ants=5,
        )

        engine_a = SimulationEngine(config=cfg)
        colony_a = Colony(colony_id=0, nest_x=4, nest_y=4)
        engine_a.add_colony(colony_a)
        engine_a.run(ticks=20)

        engine_b = SimulationEngine(config=cfg)
        colony_b = Colony(colony_id=0, nest_x=4, nest_y=4)
        engine_b.add_colony(colony_b)
        engine_b.run(ticks=20)

        # Environment state must match
        assert engine_a.environment.tick == engine_b.environment.tick
        assert engine_a.environment.time_of_day == engine_b.environment.time_of_day

        # Pheromone grids must be bit-identical
        for ptype in engine_a.pheromone_field.layers:
            assert np.array_equal(
                engine_a.pheromone_field.get_layer(ptype),
                engine_b.pheromone_field.get_layer(ptype),
            )

        # Ant positions must match
        for ant_a, ant_b in zip(
            engine_a.colonies[0].ants,
            engine_b.colonies[0].ants,
            strict=True,
        ):
            assert ant_a.x == ant_b.x
            assert ant_a.y == ant_b.y
            assert ant_a.task == ant_b.task
