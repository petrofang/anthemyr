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
        num_patches: int = 12,
        patch_radius: int = 4,
        food_per_cell: tuple[float, float] = (2.0, 5.0),
    ) -> None:
        """Place clustered food patches across the grid.

        Instead of sparse random scatter, creates a number of rich
        food patches that reward trail formation and exploitation.

        Args:
            rng: Seeded random generator.
            num_patches: Number of food patches to place.
            patch_radius: Radius of each patch in cells.
            food_per_cell: (min, max) food placed per cell in a patch.
        """
        lo, hi = food_per_cell
        for _ in range(num_patches):
            cx = int(rng.integers(0, self.width))
            cy = int(rng.integers(0, self.height))
            for dy in range(-patch_radius, patch_radius + 1):
                for dx in range(-patch_radius, patch_radius + 1):
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    # Circular falloff: cells near centre get more food
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist <= patch_radius:
                        cell = self.cells[ny][nx]
                        if not cell.is_nest:
                            cell.food += float(
                                rng.uniform(lo, hi) * (1.0 - dist / (patch_radius + 1)),
                            )

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

    def regenerate_food(
        self,
        rng: Generator,
        *,
        base_rate: float = 0.0005,
        spread_rate: float = 0.02,
        food_cap: float = 5.0,
    ) -> None:
        """Regrow food using density-dependent spread from existing patches.

        Uses a 5x5 neighbourhood (24 cells) to calculate local food
        density.  Inner-ring cells (8-connected) are weighted 1.0,
        outer-ring cells (the next 16) are weighted 0.5.  Growth
        probability scales with the *square* of normalised density,
        so:

        - Isolated food cells barely spread.
        - Cells with a moderate ring of food (8 neighbours) grow
          at a moderate rate.
        - Cells surrounded on all 24 sides thicken *rapidly*.

        This produces coherent, self-reinforcing food patches that
        grow outward from their edges while becoming denser in the
        centre — mimicking natural vegetation dynamics.

        A small ``base_rate`` allows rare spontaneous growth.

        Args:
            rng: Seeded random generator.
            base_rate: Tiny probability of spontaneous food appearance.
            spread_rate: Growth probability multiplier from neighbour
                density (applied to density²).
            food_cap: Maximum food a cell can hold.
        """
        # Snapshot current food to avoid order-dependent bias
        food_snap = [[cell.food for cell in row] for row in self.cells]

        for y, row in enumerate(self.cells):
            for x, cell in enumerate(row):
                if cell.is_nest or cell.food >= food_cap:
                    continue

                # Weighted food sum over 5x5 neighbourhood
                weighted_food = 0.0
                max_weight = 0.0
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            # Inner ring (8-connected): weight 1.0
                            # Outer ring: weight 0.5
                            w = 1.0 if abs(dx) <= 1 and abs(dy) <= 1 else 0.5
                            weighted_food += food_snap[ny][nx] * w
                            max_weight += w * food_cap

                # Normalised density 0..1
                density = weighted_food / max_weight if max_weight > 0 else 0.0

                # Quadratic scaling: dense patches grow fast,
                # sparse edges barely grow
                growth_prob = base_rate + spread_rate * density * density

                if rng.random() < growth_prob:
                    # Amount added also scales with density
                    amount = float(rng.uniform(0.1, 0.3)) + 0.3 * density
                    cell.food = min(food_cap, cell.food + amount)
