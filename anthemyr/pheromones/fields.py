"""PheromoneField — multi-layer pheromone grid.

Each pheromone type (trail, alarm, territory, etc.) is stored as a
separate NumPy 2D array.  The field provides deposit/read operations and
delegates diffusion/evaporation to ``diffusion.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class PheromoneType(Enum):
    """Distinct pheromone channels, each with its own layer."""

    TRAIL = auto()
    ALARM = auto()
    TERRITORY = auto()
    RECRUITMENT = auto()
    BROOD_CARE = auto()
    DEATH = auto()
    ROYAL = auto()


@dataclass
class PheromoneLayer:
    """A single pheromone channel stored as a 2D NumPy array.

    Attributes:
        ptype: Which pheromone this layer represents.
        grid: Concentration values (≥ 0).
        diffusion_rate: Fraction that spreads to neighbours per tick.
        evaporation_rate: Fraction lost per tick (before diffusion).
    """

    ptype: PheromoneType
    grid: NDArray[np.float64]
    diffusion_rate: float = 0.1
    evaporation_rate: float = 0.05


@dataclass
class PheromoneField:
    """All pheromone layers for a world.

    Attributes:
        width: Grid columns (must match World).
        height: Grid rows (must match World).
        layers: Mapping from PheromoneType to its layer.
    """

    width: int
    height: int
    layers: dict[PheromoneType, PheromoneLayer] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Create one layer per pheromone type, all zeroed."""
        self.layers = {}
        for ptype in PheromoneType:
            self.layers[ptype] = PheromoneLayer(
                ptype=ptype,
                grid=np.zeros((self.height, self.width), dtype=np.float64),
            )

    def deposit(self, ptype: PheromoneType, x: int, y: int, amount: float) -> None:
        """Add pheromone at a specific cell.

        Args:
            ptype: Which pheromone to deposit.
            x: Column index.
            y: Row index.
            amount: Quantity to add (must be ≥ 0).
        """
        self.layers[ptype].grid[y, x] += amount

    def read(self, ptype: PheromoneType, x: int, y: int) -> float:
        """Read pheromone concentration at a cell.

        Args:
            ptype: Which pheromone to read.
            x: Column index.
            y: Row index.

        Returns:
            Current concentration value.
        """
        return float(self.layers[ptype].grid[y, x])

    def get_layer(self, ptype: PheromoneType) -> NDArray[np.float64]:
        """Return the raw NumPy array for a pheromone layer.

        Args:
            ptype: Which pheromone type.

        Returns:
            2D array of concentration values.
        """
        return self.layers[ptype].grid
