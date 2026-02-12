"""Entry point for ``python -m anthemyr``.

Loads the default YAML config, builds a simulation engine with one
colony, and opens a Pygame window to watch the ants forage.
"""

from __future__ import annotations

import argparse
import pathlib

from anthemyr.colony.colony import Colony
from anthemyr.colony.policies import Policies
from anthemyr.colony.traits import Traits
from anthemyr.simulation.config import SimulationConfig
from anthemyr.simulation.engine import SimulationEngine
from anthemyr.ui.pygame_client import PygameRenderer

_DEFAULT_CONFIG = (
    pathlib.Path(__file__).resolve().parent.parent / "config" / "default.yaml"
)


def main() -> None:
    """Parse CLI args, create engine, launch renderer."""
    parser = argparse.ArgumentParser(
        prog="anthemyr",
        description="Anthemyr - ant superorganism simulator",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=pathlib.Path,
        default=_DEFAULT_CONFIG,
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=10,
        help="Pixel size per grid cell (default: 10)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second (default: 30)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=10.0,
        help="Simulation ticks per second (default: 10)",
    )
    args = parser.parse_args()

    config = SimulationConfig.from_yaml(args.config)
    engine = SimulationEngine(config=config)

    colony = Colony(
        colony_id=0,
        nest_x=config.world_width // 2,
        nest_y=config.world_height // 2,
        traits=Traits(),
        policies=Policies(),
    )
    engine.add_colony(colony)

    renderer = PygameRenderer(
        engine=engine,
        cell_size=args.cell_size,
        ticks_per_second=args.speed,
    )
    renderer.run(fps=args.fps)


if __name__ == "__main__":
    main()
