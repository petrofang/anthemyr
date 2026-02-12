"""Cell — a single tile in the world grid.

Each cell holds terrain properties and local resource levels.  Pheromone
concentrations are stored externally in ``PheromoneField`` layers so the
cell itself stays lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SoilType(Enum):
    """Soil composition affecting digging speed and moisture retention."""

    DIRT = "dirt"
    SAND = "sand"
    CLAY = "clay"
    ROCK = "rock"


@dataclass
class Cell:
    """A single tile in the world grid.

    Attributes:
        x: Column position.
        y: Row position.
        soil: Soil type at this location.
        moisture: Moisture level (0.0-1.0).
        temperature: Local temperature in arbitrary sim-units.
        food: Amount of harvestable food at this cell.
        is_nest: Whether this cell is part of a colony nest.
    """

    x: int
    y: int
    soil: SoilType = SoilType.DIRT
    moisture: float = 0.5
    temperature: float = 0.5
    food: float = 0.0
    is_nest: bool = False
