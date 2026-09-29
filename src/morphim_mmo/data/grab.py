"""GRAB sequence loading and subject-specific MANO hand decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from ..models.mano import load_mano_model


class _FrameCountMixin:
    """Provide frame count for motions storing per-frame global orientation."""

    global_orient: np.ndarray

    @property
    def frame_count(self) -> int:
        """Return the number of motion frames."""

        return int(self.global_orient.shape[0])


@dataclass(frozen=True)
class GrabHandMotion(_FrameCountMixin):
    """Normalized parameters for one hand in a GRAB sequence."""

    subject: str
    side: str
    framerate_hz: float
    template_path: Path
    global_orient: np.ndarray
    hand_pose: np.ndarray
    translation_m: np.ndarray

@dataclass(frozen=True)
class DecodedHand:
    """World-space MANO geometry decoded from a GRAB hand motion."""

    vertices_m: np.ndarray
    joints_m: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class GrabRigidMotion(_FrameCountMixin):
    """A static mesh and its per-frame rigid pose from GRAB."""

    name: str
    framerate_hz: float
    mesh_path: Path
    vertices_local_m: np.ndarray
    faces: np.ndarray
    global_orient: np.ndarray
    translation_m: np.ndarray

    def rotations(self) -> np.ndarray:
        """Convert per-frame axis-angle orientations to rotation matrices."""

        return Rotation.from_rotvec(self.global_orient).as_matrix()

    def transforms(self) -> np.ndarray:
        """Assemble per-frame homogeneous local-to-world transforms."""

        transforms = np.broadcast_to(np.eye(4), (self.frame_count, 4, 4)).copy()
        transforms[:, :3, :3] = self.rotations()
        transforms[:, :3, 3] = self.translation_m
        return transforms


def _resolve_reference(grab_root: Path, reference: str) -> Path:
    """Resolve a GRAB-relative path without allowing root traversal."""

    root = grab_root.resolve()
    path = (root / reference).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"GRAB reference escapes the data root: {reference}")
    return path


def load_grab_hand_motion(
    sequence_path: str | Path,
    *,
    side: str = "right",
    grab_root: str | Path | None = None,
) -> GrabHandMotion:
    """Read and validate one hand's MANO parameters from a GRAB NPZ file."""

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    sequence_path = Path(sequence_path)
    if grab_root is None:
        grab_root = sequence_path.parents[1]
    grab_root = Path(grab_root)

    with np.load(sequence_path, allow_pickle=True) as sequence:
        hand = sequence["rhand" if side == "right" else "lhand"].item()
        params = hand["params"]
        global_orient = np.asarray(params["global_orient"], dtype=np.float32)
        full_pose = np.asarray(params["fullpose"], dtype=np.float32)
        translation = np.asarray(params["transl"], dtype=np.float32)
        frame_count = int(sequence["n_frames"])
        motion = GrabHandMotion(
            subject=str(sequence["sbj_id"]),
            side=side,
            framerate_hz=float(sequence["framerate"]),
            template_path=_resolve_reference(grab_root, str(hand["vtemp"])),
            global_orient=global_orient,
            hand_pose=full_pose,
            translation_m=translation,
        )

    expected = {
        "global_orient": (frame_count, 3),
        "hand_pose": (frame_count, 45),
        "translation_m": (frame_count, 3),
    }
    for name, shape in expected.items():
        value = getattr(motion, name)
        if value.shape != shape:
            raise ValueError(f"GRAB {name} expected shape {shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"GRAB {name} contains non-finite values")
    if motion.framerate_hz <= 0:
        raise ValueError("GRAB framerate must be positive")
    if not motion.template_path.is_file():
        raise FileNotFoundError(f"Missing GRAB hand template: {motion.template_path}")
    return motion


def load_grab_rigid_motion(
    sequence_path: str | Path,
    *,
    section: str = "object",
    grab_root: str | Path | None = None,
) -> GrabRigidMotion:
    """Load the manipulated object or table mesh and per-frame rigid poses."""

    if section not in {"object", "table"}:
        raise ValueError("section must be 'object' or 'table'")
    sequence_path = Path(sequence_path)
    if grab_root is None:
        grab_root = sequence_path.parents[1]
    grab_root = Path(grab_root)
    reference_key = "object_mesh" if section == "object" else "table_mesh"
    with np.load(sequence_path, allow_pickle=True) as sequence:
        entry = sequence[section].item()
        params = entry["params"]
        global_orient = np.asarray(params["global_orient"], dtype=np.float64)
        translation = np.asarray(params["transl"], dtype=np.float64)
        frame_count = int(sequence["n_frames"])
        name = str(sequence["obj_name"]) if section == "object" else "table"
        framerate_hz = float(sequence["framerate"])
        mesh_path = _resolve_reference(grab_root, str(entry[reference_key]))

    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing GRAB {section} mesh: {mesh_path}")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"GRAB {section} vertices must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"GRAB {section} faces must have shape (F, 3)")
    if global_orient.shape != (frame_count, 3):
        raise ValueError(f"GRAB {section} global_orient has an invalid shape")
    if translation.shape != (frame_count, 3):
        raise ValueError(f"GRAB {section} translation has an invalid shape")
    if not all(np.isfinite(value).all() for value in (vertices, global_orient, translation)):
        raise ValueError(f"GRAB {section} contains non-finite values")
    return GrabRigidMotion(
        name=name,
        framerate_hz=framerate_hz,
        mesh_path=mesh_path,
        vertices_local_m=vertices,
        faces=faces,
        global_orient=global_orient,
        translation_m=translation,
    )


def decode_grab_hand(
    motion: GrabHandMotion,
    mano_model_directory: str | Path,
    *,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> DecodedHand:
    """Decode a GRAB motion using its subject-specific MANO template."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    template_mesh = trimesh.load_mesh(motion.template_path, process=False)
    template_vertices = np.asarray(template_mesh.vertices, dtype=np.float32)
    faces = np.asarray(template_mesh.faces, dtype=np.int32)
    if template_vertices.shape != (778, 3) or faces.shape != (1538, 3):
        raise ValueError(
            "GRAB MANO template must have 778 vertices and 1538 triangular faces"
        )

    device = torch.device(device)
    model = load_mano_model(
        mano_model_directory,
        motion.side,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    dtype = model.dtype
    with torch.no_grad():
        model.v_template.copy_(torch.as_tensor(template_vertices, dtype=dtype, device=device))
        vertices = []
        joints = []
        for start in range(0, motion.frame_count, batch_size):
            stop = min(start + batch_size, motion.frame_count)
            count = stop - start
            output = model(
                betas=torch.zeros((count, 10), dtype=dtype, device=device),
                global_orient=torch.as_tensor(
                    motion.global_orient[start:stop], dtype=dtype, device=device
                ),
                hand_pose=torch.as_tensor(
                    motion.hand_pose[start:stop], dtype=dtype, device=device
                ),
                transl=torch.as_tensor(
                    motion.translation_m[start:stop], dtype=dtype, device=device
                ),
                return_verts=True,
            )
            vertices.append(output.vertices.detach().cpu().numpy())
            joints.append(output.joints[:, :16].detach().cpu().numpy())

    decoded = DecodedHand(
        vertices_m=np.concatenate(vertices).astype(np.float32, copy=False),
        joints_m=np.concatenate(joints).astype(np.float32, copy=False),
        faces=faces,
    )
    if not np.isfinite(decoded.vertices_m).all() or not np.isfinite(decoded.joints_m).all():
        raise ValueError("Decoded GRAB hand contains non-finite values")
    return decoded
