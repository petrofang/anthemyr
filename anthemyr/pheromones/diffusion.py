"""Diffusion and evaporation logic for pheromone layers.

Operates on the raw NumPy arrays inside ``PheromoneLayer`` objects.
Separated from ``fields.py`` so that diffusion algorithms can be
swapped or optimised independently.
"""

from __future__ import annotations

from anthemyr.pheromones.fields import PheromoneField, PheromoneLayer


def evaporate(layer: PheromoneLayer) -> None:
    """Reduce pheromone concentrations by the layer's evaporation rate.

    Applies ``grid *= (1 - evaporation_rate)`` in-place.

    Args:
        layer: The pheromone layer to evaporate.
    """
    layer.grid *= 1.0 - layer.evaporation_rate


def diffuse(layer: PheromoneLayer) -> None:
    """Spread pheromone to neighbouring cells using a simple blur kernel.

    Each cell gives ``diffusion_rate`` of its value equally to its
    four cardinal neighbours.  Boundary cells lose less (no wrap).

    This modifies ``layer.grid`` in-place.

    Args:
        layer: The pheromone layer to diffuse.
    """
    rate = layer.diffusion_rate
    if rate <= 0:
        return

    grid = layer.grid
    donated = grid * rate
    grid *= 1.0 - rate  # keep the non-donated portion

    share = donated / 4.0
    # Shift in each cardinal direction and accumulate
    grid[1:, :] += share[:-1, :]  # donate downward
    grid[:-1, :] += share[1:, :]  # donate upward
    grid[:, 1:] += share[:, :-1]  # donate rightward
    grid[:, :-1] += share[:, 1:]  # donate leftward


def update_field(field: PheromoneField) -> None:
    """Run one tick of evaporation + diffusion on all layers.

    TRAIL pheromone skips diffusion entirely -- it should fade in
    place (evaporate) rather than spreading into an amorphous cloud.
    This preserves the linear trail shape that carrying ants deposit,
    creating clear directional signals rather than noisy blobs.

    All other pheromone types receive both evaporation and diffusion.

    Args:
        field: The complete pheromone field to update.
    """
    from anthemyr.pheromones.fields import PheromoneType

    for layer in field.layers.values():
        evaporate(layer)
        if layer.ptype is not PheromoneType.TRAIL:
            diffuse(layer)
