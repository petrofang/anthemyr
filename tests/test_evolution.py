"""Tests for anthemyr.colony.evolution (stub — Phase 5)."""

from anthemyr.colony.evolution import evolve_traits
from anthemyr.colony.traits import Traits


class TestEvolution:
    """Tests for generational trait evolution."""

    def test_evolve_preserves_traits_stub(self) -> None:
        """Until real evolution logic is implemented, traits pass through."""
        parent = Traits(aggression=0.8, exploration=0.2)
        child = evolve_traits(
            parent,
            survival_ticks=500,
            food_efficiency=1.2,
            war_success=0.6,
        )
        assert child.aggression == parent.aggression
        assert child.exploration == parent.exploration
