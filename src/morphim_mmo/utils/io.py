"""Consistent filesystem I/O for command-line workflows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping and reject non-mapping documents."""

    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object and reject non-object documents."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create the parent directory and write a readable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_npz(path: Path, **arrays: Any) -> None:
    """Create the parent directory and save a compressed NumPy archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_image(image: Any, path: Path, *, quality: int = 96) -> None:
    """Create the parent directory and save a Pillow-compatible image."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=quality)


def save_gif(frames: Sequence[Any], path: Path, *, fps: float) -> None:
    """Save equally timed Pillow-compatible frames as a looping GIF."""

    if not frames:
        raise ValueError("at least one GIF frame is required")
    if fps <= 0:
        raise ValueError("fps must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=round(1000.0 / fps),
        loop=0,
        disposal=2,
        optimize=False,
    )
