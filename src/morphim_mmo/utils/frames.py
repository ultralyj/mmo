"""Parsing and validation of command-line frame selections."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def parse_frame_indices(
    specification: str,
    frame_count: int,
    *,
    aliases: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Parse ``all``, an alias, CSV indices, or ``START:STOP[:STRIDE]``.

    Returned indices are guaranteed to be non-empty, in range, and strictly
    increasing, which is required by the sequential warm-start optimizers.
    """

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if specification == "all":
        frames = np.arange(frame_count, dtype=np.int32)
    elif aliases is not None and specification in aliases:
        frames = np.asarray([aliases[specification]], dtype=np.int32)
    elif ":" in specification:
        parts = specification.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("frame range must be START:STOP or START:STOP:STRIDE")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else frame_count
        stride = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if stride <= 0:
            raise ValueError("frame stride must be positive")
        frames = np.arange(start, stop, stride, dtype=np.int32)
    else:
        frames = np.asarray(
            [int(value) for value in specification.split(",")], dtype=np.int32
        )
    if frames.size == 0:
        raise ValueError("no frames selected")
    if np.any(frames < 0) or np.any(frames >= frame_count):
        raise ValueError(f"frame selection must be within [0, {frame_count})")
    if np.any(np.diff(frames) <= 0):
        raise ValueError("frames must be strictly increasing")
    return frames

