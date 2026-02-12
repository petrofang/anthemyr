"""Pygame 2D visualization for the Anthemyr simulation.

Renders the world grid, food, pheromone trails, and ants in a window.
The simulation steps at a configurable tick rate while the display
refreshes at the Pygame frame rate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pygame

if TYPE_CHECKING:
    from anthemyr.simulation.engine import SimulationEngine

from anthemyr.colony.ant import Task
from anthemyr.pheromones.fields import PheromoneType

# Colour palette
_BG = (30, 20, 10)
_NEST = (80, 60, 40)
_GRID_LINE = (40, 30, 20)

# Ant colours by task
_ANT_COLOURS: dict[Task, tuple[int, int, int]] = {
    Task.IDLE: (180, 180, 180),
    Task.FORAGING: (100, 200, 100),
    Task.CARRYING_FOOD: (255, 200, 50),
    Task.BROOD_CARE: (200, 150, 255),
    Task.PATROLLING: (100, 150, 255),
    Task.FIGHTING: (255, 80, 80),
    Task.WASTE_MANAGEMENT: (150, 120, 80),
}

# Food colour range (dark green -> bright green)
_FOOD_LO = np.array([20, 60, 10], dtype=np.float64)
_FOOD_HI = np.array([50, 200, 30], dtype=np.float64)

# Trail pheromone colour (cyan glow)
_TRAIL_COLOUR = np.array([0, 180, 255], dtype=np.float64)


class PygameRenderer:
    """Renders a SimulationEngine state into a Pygame window.

    Attributes:
        engine: The simulation engine to visualise.
        cell_size: Pixel size of each grid cell.
        screen: The Pygame display surface.
    """

    # Speed presets: ticks per second at 30 fps
    _SPEED_STEPS: ClassVar[list[float]] = [
        0.5,
        1.0,
        3.0,
        5.0,
        10.0,
        15.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
    ]

    def __init__(
        self,
        engine: SimulationEngine,
        cell_size: int = 10,
        ticks_per_second: float = 10.0,
    ) -> None:
        """Initialise the renderer.

        Args:
            engine: The simulation engine to render.
            cell_size: Pixel width/height per grid cell.
            ticks_per_second: Simulation ticks per real-time second.
        """
        self.engine = engine
        self.cell_size = cell_size
        self.ticks_per_second = ticks_per_second
        self._speed_index = self._nearest_speed(ticks_per_second)
        self._tick_accumulator = 0.0

        w = engine.world.width * cell_size
        h = engine.world.height * cell_size
        self._panel_width = 220
        self._win_w = w + self._panel_width
        self._win_h = h

        pygame.init()
        self.screen = pygame.display.set_mode((self._win_w, self._win_h))
        pygame.display.set_caption("Anthemyr")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.running = True
        self.paused = False

    def _nearest_speed(self, tps: float) -> int:
        """Return the index of the closest speed preset."""
        best = 0
        best_diff = abs(self._SPEED_STEPS[0] - tps)
        for i, s in enumerate(self._SPEED_STEPS):
            diff = abs(s - tps)
            if diff < best_diff:
                best, best_diff = i, diff
        return best

    def run(self, fps: int = 30) -> None:
        """Main loop: handle events, step sim, render.

        Args:
            fps: Target frames per second.
        """
        while self.running:
            dt = self.clock.tick(fps) / 1000.0  # seconds elapsed
            self._handle_events()
            if not self.paused:
                self._tick_accumulator += self.ticks_per_second * dt
                steps = int(self._tick_accumulator)
                self._tick_accumulator -= steps
                for _ in range(steps):
                    self.engine.step()
            self._draw()

        pygame.quit()

    def _handle_events(self) -> None:
        """Process Pygame input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self._speed_index = min(
                        len(self._SPEED_STEPS) - 1,
                        self._speed_index + 1,
                    )
                    self.ticks_per_second = self._SPEED_STEPS[self._speed_index]
                elif event.key == pygame.K_MINUS:
                    self._speed_index = max(0, self._speed_index - 1)
                    self.ticks_per_second = self._SPEED_STEPS[self._speed_index]

    def _draw(self) -> None:
        """Render one frame."""
        self.screen.fill(_BG)
        self._draw_terrain()
        self._draw_trail_overlay()
        self._draw_food()
        self._draw_ants()
        self._draw_info_panel()
        pygame.display.flip()

    def _draw_terrain(self) -> None:
        """Draw nest cells as a distinct colour."""
        cs = self.cell_size
        world = self.engine.world
        for y in range(world.height):
            for x in range(world.width):
                cell = world.cells[y][x]
                if cell.is_nest:
                    pygame.draw.rect(
                        self.screen,
                        _NEST,
                        (x * cs, y * cs, cs, cs),
                    )

    def _draw_food(self) -> None:
        """Draw food as green squares scaled by amount."""
        cs = self.cell_size
        world = self.engine.world
        for y in range(world.height):
            for x in range(world.width):
                food = world.cells[y][x].food
                if food > 0:
                    t = min(food / 5.0, 1.0)
                    colour = _FOOD_LO + t * (_FOOD_HI - _FOOD_LO)
                    pygame.draw.rect(
                        self.screen,
                        colour.astype(int).tolist(),
                        (x * cs, y * cs, cs, cs),
                    )

    def _draw_trail_overlay(self) -> None:
        """Draw trail pheromone as a translucent cyan overlay."""
        cs = self.cell_size
        trail = self.engine.pheromone_field.get_layer(PheromoneType.TRAIL)
        max_val = trail.max()
        if max_val <= 0:
            return

        overlay = pygame.Surface(
            (self.engine.world.width * cs, self.engine.world.height * cs),
            pygame.SRCALPHA,
        )

        for y in range(trail.shape[0]):
            for x in range(trail.shape[1]):
                val = trail[y, x]
                if val > 0.01:
                    alpha = int(min(val / max_val, 1.0) * 120)
                    colour = _TRAIL_COLOUR.astype(int).tolist()
                    pygame.draw.rect(
                        overlay,
                        (*colour, alpha),
                        (x * cs, y * cs, cs, cs),
                    )

        self.screen.blit(overlay, (0, 0))

    def _draw_ants(self) -> None:
        """Draw each ant as a small coloured dot."""
        cs = self.cell_size
        radius = max(2, cs // 3)
        for colony in self.engine.colonies:
            for ant in colony.ants:
                colour = _ANT_COLOURS.get(ant.task, (200, 200, 200))
                cx = ant.x * cs + cs // 2
                cy = ant.y * cs + cs // 2
                pygame.draw.circle(self.screen, colour, (cx, cy), radius)

    def _draw_info_panel(self) -> None:
        """Draw a stats panel on the right side of the window."""
        panel_x = self.engine.world.width * self.cell_size + 10
        y = 10

        lines = [
            f"Tick: {self.engine.tick}",
            f"Speed: {self.ticks_per_second:.1f} t/s",
            f"{'PAUSED' if self.paused else 'RUNNING'}",
            "",
            "--- Colony ---",
        ]

        for colony in self.engine.colonies:
            task_counts: dict[str, int] = {}
            for ant in colony.ants:
                name = ant.task.name
                task_counts[name] = task_counts.get(name, 0) + 1

            lines += [
                f"Ants: {len(colony.ants)}",
                f"Food: {colony.food_stores:.1f}",
                f"Brood: {colony.brood_count}",
                "",
            ]
            for task_name, count in sorted(task_counts.items()):
                lines.append(f"  {task_name}: {count}")

        lines += [
            "",
            "--- Controls ---",
            "SPACE: pause",
            "+/-: speed",
            "ESC: quit",
        ]

        for line in lines:
            surf = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(surf, (panel_x, y))
            y += 18
