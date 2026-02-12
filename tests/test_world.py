"""Tests for anthemyr.world.world and anthemyr.world.cell."""

from anthemyr.world.cell import Cell, SoilType
from anthemyr.world.world import World


class TestCell:
    """Tests for the Cell dataclass."""

    def test_default_values(self) -> None:
        cell = Cell(x=0, y=0)
        assert cell.soil == SoilType.DIRT
        assert cell.moisture == 0.5
        assert cell.food == 0.0
        assert cell.is_nest is False


class TestWorld:
    """Tests for the World grid."""

    def test_dimensions(self, small_world: World) -> None:
        assert small_world.width == 8
        assert small_world.height == 8
        assert len(small_world.cells) == 8
        assert len(small_world.cells[0]) == 8

    def test_cell_at_valid(self, small_world: World) -> None:
        cell = small_world.cell_at(3, 5)
        assert cell.x == 3
        assert cell.y == 5

    def test_cell_at_out_of_bounds(self, small_world: World) -> None:
        import pytest

        with pytest.raises(IndexError):
            small_world.cell_at(8, 0)

    def test_neighbours_corner(self, small_world: World) -> None:
        # Top-left corner — should have 3 diagonal neighbours
        neighbours = small_world.neighbours(0, 0, include_diagonals=True)
        assert len(neighbours) == 3

    def test_neighbours_cardinal_only(self, small_world: World) -> None:
        # Center-ish cell, cardinal only
        neighbours = small_world.neighbours(3, 3, include_diagonals=False)
        assert len(neighbours) == 4

    def test_neighbours_center(self, small_world: World) -> None:
        neighbours = small_world.neighbours(3, 3, include_diagonals=True)
        assert len(neighbours) == 8
