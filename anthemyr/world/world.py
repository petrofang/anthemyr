"""World grid — the spatial container for the simulation.

The World owns cells arranged in a 2D grid and provides spatial queries
(neighbours, line-of-sight, region iteration) used by ants, pheromone
diffusion, and environment updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

from anthemyr.world.cell import Cell


@dataclass
class World:
    """A 2D grid world that contains all spatial simulation state.

    Attributes:
        width: Number of columns in the grid.
        height: Number of rows in the grid.
        cells: 2D list of Cell objects indexed as ``cells[y][x]``.
    """

    width: int
    height: int
    cells: list[list[Cell]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise the grid with default cells."""
        self.cells = [
            [Cell(x=x, y=y) for x in range(self.width)] for y in range(self.height)
        ]

    def cell_at(self, x: int, y: int) -> Cell:
        """Return the cell at grid coordinates ``(x, y)``.

        Args:
            x: Column index.
            y: Row index.

        Raises:
            IndexError: If coordinates are out of bounds.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            msg = f"({x}, {y}) out of bounds for {self.width}x{self.height}"
            raise IndexError(msg)
        return self.cells[y][x]

    def neighbours(
        self,
        x: int,
        y: int,
        *,
        include_diagonals: bool = True,
    ) -> list[Cell]:
        """Return adjacent cells for the given position.

        Args:
            x: Column index.
            y: Row index.
            include_diagonals: If True, return up to 8 neighbours; otherwise 4.

        Returns:
            List of neighbouring Cell objects (excludes out-of-bounds).
        """
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if include_diagonals:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        result: list[Cell] = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                result.append(self.cells[ny][nx])
        return result

    def populate(
        self,
        rng: Generator,
        *,
        food_density: float = 0.15,
        food_amount_range: tuple[float, float] = (1.0, 5.0),
    ) -> None:
        """Scatter food across the grid randomly.

        Args:
            rng: Seeded random generator.
            food_density: Probability that any given cell has food.
            food_amount_range: (min, max) food placed per seeded cell.
        """
        lo, hi = food_amount_range
        for row in self.cells:
            for cell in row:
                if rng.random() < food_density:
                    cell.food = float(rng.uniform(lo, hi))

    def mark_nest(self, cx: int, cy: int, radius: int = 2) -> None:
        """Mark cells around ``(cx, cy)`` as nest territory.

        Args:
            cx: Centre column of the nest.
            cy: Centre row of the nest.
            radius: How many cells outward to mark.
        """
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.cells[ny][nx].is_nest = True
                    self.cells[ny][nx].food = 0.0
