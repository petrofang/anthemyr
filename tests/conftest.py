"""Shared fixtures for the Anthemyr test suite."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from anthemyr.colony.colony import Colony
from anthemyr.pheromones.fields import PheromoneField
from anthemyr.simulation.config import SimulationConfig
from anthemyr.world.world import World


@pytest.fixture
def rng() -> Generator:
    """A deterministic random generator for reproducible tests."""
    return np.random.default_rng(seed=12345)


@pytest.fixture
def small_world() -> World:
    """A small 8x8 world for fast tests."""
    return World(width=8, height=8)


@pytest.fixture
def default_config() -> SimulationConfig:
    """Default simulation config (no YAML file needed)."""
    return SimulationConfig()


@pytest.fixture
def default_colony(rng: Generator) -> Colony:
    """A colony at (4, 4) with default traits, pre-populated with 5 ants."""
    colony = Colony(colony_id=0, nest_x=4, nest_y=4)
    for _ in range(5):
        colony.spawn_ant(rng)
    return colony


@pytest.fixture
def small_pheromone_field() -> PheromoneField:
    """An 8x8 pheromone field for fast tests."""
    return PheromoneField(width=8, height=8)
