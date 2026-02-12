"""Smoke tests for the UI module (no display required)."""

from __future__ import annotations

from anthemyr.ui.pygame_client import PygameRenderer


def test_pygame_renderer_importable() -> None:
    """PygameRenderer class is importable without initialising pygame."""
    assert PygameRenderer is not None


def test_main_module_importable() -> None:
    """The __main__ module is importable and exposes main()."""
    from anthemyr.__main__ import main

    assert callable(main)
