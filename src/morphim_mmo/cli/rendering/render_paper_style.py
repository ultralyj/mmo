"""Smooth, paper-style rendering of Sharpa morphology matching."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...optimization.alignment import palm_frame
from ...utils.io import load_yaml, save_image
from ...visualization.common import (
    BACKGROUND,
    draw_centered_text,
    load_font,
    render_meshes,
)
from ...visualization.urdf_geometry import load_urdf_visual_geometry


SKIN = (0.78, 0.55, 0.42, 1.0)
ROBOT = (0.80, 0.86, 0.90, 1.0)


def main() -> None:
    """Render smooth morphology-initialization, alignment, and overlay panels."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sharpa_wave_right.yaml"))
    parser.add_argument("--geometry", type=Path, default=Path("outputs/morphology_match.npz"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/morphology_paper_style.png")
    )
    parser.add_argument(
        "--aligned-output",
        type=Path,
        default=Path("outputs/morphology_aligned_mano_smooth.png"),
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    robot_vertices, robot_faces = load_urdf_visual_geometry(Path(config["urdf"]))
    with np.load(args.geometry, allow_pickle=False) as data:
        initial_vertices = data["initial_vertices_m"]
        optimized_vertices = data["optimized_vertices_m"]
        mano_faces = data["mano_faces"]
        optimized_joints = data["optimized_joints_m"]
        tip_indices = data["mano_tip_vertex_indices"]

    frame = palm_frame(
        optimized_joints[0],
        optimized_vertices[tip_indices[0]],
        optimized_vertices[tip_indices[2]],
    )
    lateral, longitudinal, normal = frame.T
    palm = normal
    oblique = normal - 0.48 * lateral + 0.24 * longitudinal
    oblique /= np.linalg.norm(oblique)
    side = -lateral + 0.08 * normal
    side /= np.linalg.norm(side)
    cell_size = (320, 390)
    initial_palm = render_meshes(
        [(initial_vertices, mano_faces, SKIN)],
        palm,
        cell_size,
        camera_up=longitudinal,
        fit_aspect=True,
        camera_scale=0.61,
    )
    initial_oblique = render_meshes(
        [(initial_vertices, mano_faces, SKIN)],
        oblique,
        cell_size,
        camera_up=longitudinal,
        fit_aspect=True,
        camera_scale=0.61,
    )
    aligned_palm = render_meshes(
        [(optimized_vertices, mano_faces, SKIN)],
        palm,
        cell_size,
        camera_up=longitudinal,
        fit_aspect=True,
        camera_scale=0.61,
    )
    aligned_oblique = render_meshes(
        [(optimized_vertices, mano_faces, SKIN)],
        oblique,
        cell_size,
        camera_up=longitudinal,
        fit_aspect=True,
        camera_scale=0.61,
    )
    overlay = render_meshes(
        [
            (optimized_vertices, mano_faces, SKIN),
            (robot_vertices, robot_faces, ROBOT),
        ],
        oblique,
        (390, 390),
        camera_up=longitudinal,
        fit_aspect=True,
        camera_scale=0.61,
    )

    canvas = Image.new("RGBA", (1770, 530), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw_centered_text(
        draw, (25, 18, 640, 75), "Initialize Scaled MANO", load_font(29, bold=True)
    )
    draw_centered_text(
        draw, (790, 18, 1370, 75), "Morphology Aligned MANO", load_font(29, bold=True)
    )
    draw_centered_text(
        draw, (1425, 18, 1740, 90), "Overlay with\nTarget Robot", load_font(27, bold=True)
    )
    canvas.alpha_composite(initial_palm, (20, 86))
    canvas.alpha_composite(initial_oblique, (315, 86))
    canvas.alpha_composite(aligned_palm, (765, 86))
    canvas.alpha_composite(aligned_oblique, (1060, 86))
    canvas.alpha_composite(overlay, (1380, 86))
    draw.line((650, 280, 750, 280), fill=(75, 78, 80, 255), width=5)
    draw.polygon([(750, 280), (730, 269), (730, 291)], fill=(75, 78, 80, 255))
    draw_centered_text(
        draw, (630, 308, 770, 350), "Optimization", load_font(21, bold=True)
    )
    draw.line((1378, 65, 1378, 485), fill=(170, 174, 177, 255), width=3)
    save_image(canvas.convert("RGB"), args.output)

    aligned = Image.new("RGBA", (1240, 475), BACKGROUND)
    aligned.alpha_composite(
        render_meshes(
            [(optimized_vertices, mano_faces, SKIN)],
            palm,
            (400, 400),
            camera_up=longitudinal,
            fit_aspect=True,
            camera_scale=0.61,
        ),
        (15, 60),
    )
    aligned.alpha_composite(
        render_meshes(
            [(optimized_vertices, mano_faces, SKIN)],
            oblique,
            (400, 400),
            camera_up=longitudinal,
            fit_aspect=True,
            camera_scale=0.61,
        ),
        (420, 60),
    )
    aligned.alpha_composite(
        render_meshes(
            [(optimized_vertices, mano_faces, SKIN)],
            side,
            (400, 400),
            camera_up=longitudinal,
            fit_aspect=True,
            camera_scale=0.61,
        ),
        (825, 60),
    )
    aligned_draw = ImageDraw.Draw(aligned)
    draw_centered_text(
        aligned_draw,
        (0, 12, 1240, 55),
        "Morphology Aligned MANO — Smooth Surface",
        load_font(27, bold=True),
    )
    save_image(aligned.convert("RGB"), args.aligned_output)
    print(args.output)
    print(args.aligned_output)


if __name__ == "__main__":
    main()
