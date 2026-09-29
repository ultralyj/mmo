"""Shared headless rendering and image-annotation primitives."""

from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender
import trimesh


BACKGROUND = (239, 241, 242, 255)
TEXT_COLOR = (35, 38, 40, 255)
CONTACT_COLOR = (0.88, 0.18, 0.12, 1.0)

MeshSpec = tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]


def look_at(
    center: np.ndarray,
    direction: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a camera pose looking at ``center`` from ``direction``."""

    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    up = np.asarray([0.0, 0.0, 1.0] if up is None else up, dtype=float)
    eye = np.asarray(center) + direction
    forward = (np.asarray(center) - eye) / np.linalg.norm(np.asarray(center) - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = camera_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def contact_spheres(points: np.ndarray, radius: float = 0.0015) -> trimesh.Trimesh:
    """Create one low-poly marker sphere at each contact point."""

    spheres = []
    for point in np.asarray(points):
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        sphere.apply_translation(point)
        spheres.append(sphere)
    if not spheres:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(spheres)


def render_meshes(
    meshes: list[MeshSpec],
    direction: tuple[float, float, float],
    size: tuple[int, int],
    *,
    camera_up: np.ndarray | None = None,
    contact_points: np.ndarray | None = None,
    fit_aspect: bool = False,
    camera_scale: float = 0.62,
    ambient_intensity: float = 0.46,
    key_intensity: float = 2.4,
    fill_intensity: float = 1.15,
    fill_direction: tuple[float, float, float] = (-0.6, -0.3, 0.8),
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> Image.Image:
    """Render colored triangle meshes with a consistent orthographic setup."""

    if not meshes:
        raise ValueError("at least one mesh is required")
    if bounds is None:
        all_vertices = np.concatenate([vertices for vertices, _, _ in meshes])
        lower, upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
    else:
        lower, upper = (np.asarray(value, dtype=float) for value in bounds)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("bounds must contain increasing xyz vectors")
    center = (lower + upper) / 2.0
    extent = float(np.max(upper - lower))
    scene = pyrender.Scene(
        bg_color=np.asarray(BACKGROUND, dtype=np.uint8),
        ambient_light=np.full(3, ambient_intensity),
    )
    for vertices, faces, color in meshes:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.72,
            alphaMode="BLEND" if color[3] < 1.0 else "OPAQUE",
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
    if contact_points is not None and len(contact_points):
        contact_mesh = contact_spheres(contact_points)
        contact_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=CONTACT_COLOR,
            metallicFactor=0.0,
            roughnessFactor=0.58,
        )
        scene.add(
            pyrender.Mesh.from_trimesh(
                contact_mesh, material=contact_material, smooth=True
            )
        )

    camera_pose = look_at(center, np.asarray(direction), camera_up)
    aspect_scale = max(size[0] / size[1], 1.0) if fit_aspect else 1.0
    scene.add(
        pyrender.OrthographicCamera(
            xmag=extent * camera_scale * aspect_scale,
            ymag=extent * camera_scale,
        ),
        pose=camera_pose,
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=key_intensity),
        pose=camera_pose,
    )
    fill_pose = look_at(center, np.asarray(fill_direction))
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=fill_intensity),
        pose=fill_pose,
    )
    renderer = pyrender.OffscreenRenderer(*size)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return Image.fromarray(color, mode="RGBA")


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load the bundled-system DejaVu font used by all result figures."""

    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(filename, size=size)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Draw multiline text horizontally centered inside a bounding box."""

    bounds = draw.multiline_textbbox(
        (0, 0), text, font=font, align="center", spacing=4
    )
    width = bounds[2] - bounds[0]
    x = box[0] + (box[2] - box[0] - width) / 2
    draw.multiline_text(
        (x, box[1]),
        text,
        font=font,
        fill=TEXT_COLOR,
        align="center",
        spacing=4,
    )
