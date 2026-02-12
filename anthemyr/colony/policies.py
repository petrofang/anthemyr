"""Policies — player-adjustable colony-wide doctrine sliders.

Policies are *short-term* knobs the player tweaks during gameplay.  They
shift the effective threshold distributions and pheromone sensitivities
without changing the underlying genetics (Traits).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Policies:
    """Active policy-slider settings for a colony.

    All values are normalised 0.0-1.0.

    Attributes:
        aggression: Willingness to fight vs. flee.
        exploration: Preference for scouting new territory vs. exploiting known.
        sanitation: Priority given to waste management and corpse removal.
        foraging_radius: How far foragers range from the nest.
        brood_priority: Relative investment in brood care vs. other tasks.
        caste_soldier_ratio: Target proportion of soldier-caste ants.
    """

    aggression: float = 0.3
    exploration: float = 0.5
    sanitation: float = 0.5
    foraging_radius: float = 0.5
    brood_priority: float = 0.5
    caste_soldier_ratio: float = 0.2
