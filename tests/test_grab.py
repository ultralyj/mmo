from pathlib import Path

import numpy as np
import pytest

from morphim_mmo.data.grab import _resolve_reference


def test_resolve_reference_stays_under_grab_root(tmp_path: Path):
    expected = tmp_path / "tools" / "mesh.ply"
    assert _resolve_reference(tmp_path, "tools/mesh.ply") == expected


def test_resolve_reference_rejects_parent_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="escapes"):
        _resolve_reference(tmp_path, "../secret.ply")


def test_timestamp_convention():
    rate = 120.0
    np.testing.assert_allclose(np.arange(3) / rate, [0.0, 1 / 120, 2 / 120])
