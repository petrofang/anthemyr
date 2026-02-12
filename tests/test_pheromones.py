"""Tests for anthemyr.pheromones — fields, diffusion, evaporation."""

import numpy as np

from anthemyr.pheromones.diffusion import diffuse, evaporate
from anthemyr.pheromones.fields import (
    PheromoneField,
    PheromoneLayer,
    PheromoneType,
)


class TestPheromoneField:
    """Tests for PheromoneField setup and basic operations."""

    def test_all_layers_created(self, small_pheromone_field: PheromoneField) -> None:
        for ptype in PheromoneType:
            assert ptype in small_pheromone_field.layers

    def test_initial_concentrations_zero(
        self,
        small_pheromone_field: PheromoneField,
    ) -> None:
        for layer in small_pheromone_field.layers.values():
            assert np.all(layer.grid == 0.0)

    def test_deposit_and_read(self, small_pheromone_field: PheromoneField) -> None:
        small_pheromone_field.deposit(PheromoneType.TRAIL, x=3, y=4, amount=1.5)
        assert small_pheromone_field.read(PheromoneType.TRAIL, x=3, y=4) == 1.5

    def test_deposit_accumulates(self, small_pheromone_field: PheromoneField) -> None:
        small_pheromone_field.deposit(PheromoneType.ALARM, x=1, y=1, amount=1.0)
        small_pheromone_field.deposit(PheromoneType.ALARM, x=1, y=1, amount=0.5)
        assert small_pheromone_field.read(PheromoneType.ALARM, x=1, y=1) == 1.5


class TestEvaporation:
    """Tests for pheromone evaporation."""

    def test_evaporation_reduces_concentration(self) -> None:
        grid = np.ones((4, 4), dtype=np.float64)
        layer = PheromoneLayer(
            ptype=PheromoneType.TRAIL,
            grid=grid,
            evaporation_rate=0.1,
        )
        evaporate(layer)
        assert np.allclose(layer.grid, 0.9)

    def test_zero_evaporation(self) -> None:
        grid = np.ones((4, 4), dtype=np.float64)
        layer = PheromoneLayer(
            ptype=PheromoneType.TRAIL,
            grid=grid,
            evaporation_rate=0.0,
        )
        evaporate(layer)
        assert np.allclose(layer.grid, 1.0)


class TestDiffusion:
    """Tests for pheromone diffusion."""

    def test_total_concentration_conserved(self) -> None:
        """Diffusion should not create or destroy pheromone."""
        grid = np.zeros((8, 8), dtype=np.float64)
        grid[4, 4] = 10.0
        total_before = grid.sum()

        layer = PheromoneLayer(
            ptype=PheromoneType.TRAIL,
            grid=grid,
            diffusion_rate=0.2,
            evaporation_rate=0.0,
        )
        diffuse(layer)
        total_after = layer.grid.sum()
        assert np.isclose(total_before, total_after), (
            f"Diffusion changed total: {total_before} → {total_after}"
        )

    def test_diffusion_spreads(self) -> None:
        """After diffusion, neighbours of a point source should be non-zero."""
        grid = np.zeros((8, 8), dtype=np.float64)
        grid[4, 4] = 10.0
        layer = PheromoneLayer(
            ptype=PheromoneType.TRAIL,
            grid=grid,
            diffusion_rate=0.2,
            evaporation_rate=0.0,
        )
        diffuse(layer)
        # Cardinal neighbours should have received pheromone
        assert layer.grid[3, 4] > 0
        assert layer.grid[5, 4] > 0
        assert layer.grid[4, 3] > 0
        assert layer.grid[4, 5] > 0
