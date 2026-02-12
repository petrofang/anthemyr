"""Environment — weather, day/night, predators, and resource regeneration.

Updated first in each simulation tick so that ants and pheromones react
to the current environmental state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

    from anthemyr.world.world import World


@dataclass
class Environment:
    """Global environmental state that changes each tick.

    Attributes:
        tick: Current simulation tick.
        time_of_day: Normalised time (0.0 = midnight, 0.5 = noon).
        rain_intensity: Current rain level (0.0-1.0).
        day_length: Number of ticks in a full day/night cycle.
    """

    tick: int = 0
    time_of_day: float = 0.0
    rain_intensity: float = 0.0
    day_length: int = 100

    @property
    def is_daytime(self) -> bool:
        """Return True if the current time is between dawn and dusk."""
        return 0.25 <= self.time_of_day < 0.75

    def update(self, world: World, rng: Generator) -> None:
        """Advance the environment by one tick.

        Updates time-of-day, weather, and may spawn or move predators.

        Args:
            world: The world grid (for spatially-varying effects).
            rng: Seeded random generator.
        """
        self.tick += 1
        self.time_of_day = (self.tick % self.day_length) / self.day_length

        # Simple stochastic rain model — replace with config-driven weather later
        if rng.random() < 0.01:
            self.rain_intensity = float(rng.uniform(0.2, 1.0))
        elif self.rain_intensity > 0:
            self.rain_intensity = max(0.0, self.rain_intensity - 0.05)
