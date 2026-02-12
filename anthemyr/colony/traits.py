"""Traits — heritable genetic parameters for a colony lineage.

Traits define the *default distributions* from which individual ant
thresholds are sampled.  They evolve across generations during the
mating-flight phase and represent long-term adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Traits:
    """Colony-wide genetic trait profile.

    All threshold means are in arbitrary stimulus-units.  Lower means
    → ants respond more readily to that stimulus.

    Attributes:
        foraging_threshold_mean: Mean threshold for food-seeking behaviour.
        alarm_threshold_mean: Mean threshold for alarm/defense response.
        brood_care_threshold_mean: Mean threshold for brood-tending.
        waste_threshold_mean: Mean threshold for waste management.
        threshold_variance: Shared variance applied when sampling individual
            ant thresholds - higher variance means more behavioural diversity.
        aggression: Innate aggression bias (0.0-1.0).
        exploration: Tendency to explore vs. exploit (0.0-1.0).
        disease_resistance: Baseline resistance to pathogens (0.0-1.0).
    """

    foraging_threshold_mean: float = 0.5
    alarm_threshold_mean: float = 0.5
    brood_care_threshold_mean: float = 0.5
    waste_threshold_mean: float = 0.5
    threshold_variance: float = 0.15

    aggression: float = 0.3
    exploration: float = 0.5
    disease_resistance: float = 0.5
