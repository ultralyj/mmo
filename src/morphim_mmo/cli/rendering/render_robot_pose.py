"""Render contact-matched MANO, recovered Wuji pose, and their overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import numpy as np
from PIL import Image, ImageDraw

from ...utils.io import load_yaml, save_image
from ...visualization.common import (
    BACKGROUND,
    draw_centered_text,
    load_font,
    render_meshes,
)
from ...visualization.urdf_geometry import load_urdf_visual_geometry
from .render_contact_matching import OBJECT, SKIN


ROBOT = (0.37, 0.65, 0.82, 1.0)
MANO_GHOST = (0.78, 0.55, 0.42, 0.28)


def main() -> None:
    """Render contact-matched MANO, recovered robot pose, and their overlay."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/wuji_hand2_beta2_right.yaml")
    )
    parser.add_argument(
        "--robot-pose", type=Path, default=Path("outputs/wuji_hand2/robot_pose_full.npz")
    )
    parser.add_argument(
        "--contact-match",
        type=Path,
        default=Path("outputs/wuji_hand2/contact_match_full.npz"),
    )
    parser.add_argument(
        "--contacts", type=Path, default=Path("../data/processed/s1/apple_lift_contacts.npz")
    )
    parser.add_argument("--frame", type=int, default=1732)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wuji_hand2/robot_pose_recovery.png")
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    robot_result = np.load(args.robot_pose, allow_pickle=False)
    contact_result = np.load(args.contact_match, allow_pickle=False)
    row_matches = np.flatnonzero(robot_result["frame_index"] == args.frame)
    contact_matches = np.flatnonzero(contact_result["frame_index"] == args.frame)
    if len(row_matches) != 1 or len(contact_matches) != 1:
        raise ValueError("requested frame must be present exactly once in both results")
    row = int(row_matches[0])
    contact_row = int(contact_matches[0])
    joint_names = [str(value) for value in robot_result["joint_names"]]
    configuration = dict(
        zip(joint_names, robot_result["joint_positions_rad"][row], strict=True)
    )
    base_transform = np.asarray(
        jaxlie.SE3(jnp.asarray(robot_result["base_wxyz_xyz"][row])).as_matrix()
    )
    robot_vertices, robot_faces = load_urdf_visual_geometry(
        Path(config["urdf"]), configuration, base_transform
    )
    mano_vertices = contact_result["optimized_vertices_m"][contact_row]
    mano_faces = contact_result["mano_faces"]

    contacts = np.load(args.contacts, allow_pickle=False)
    object_transform = contacts["object_to_world"][args.frame]
    object_vertices = (
        contacts["object_vertices_local_m"] @ object_transform[:3, :3].T
        + object_transform[:3, 3]
    )
    object_faces = contacts["object_faces"]
    offsets = contacts["contact_offsets"]
    start, stop = int(offsets[args.frame]), int(offsets[args.frame + 1])
    contact_indices = contacts["contact_indices"][start:stop]
    contact_points = mano_vertices[contact_indices]

    direction = (1.0, -1.2, 0.72)
    size = (470, 440)
    panels = [
        render_meshes(
            [(object_vertices, object_faces, OBJECT), (mano_vertices, mano_faces, SKIN)],
            direction,
            size,
            contact_points=contact_points,
            ambient_intensity=0.48,
            key_intensity=2.5,
            fill_intensity=1.1,
            fill_direction=(-0.7, 0.3, 0.8),
        ),
        render_meshes(
            [(object_vertices, object_faces, OBJECT), (robot_vertices, robot_faces, ROBOT)],
            direction,
            size,
            contact_points=contact_points,
            ambient_intensity=0.48,
            key_intensity=2.5,
            fill_intensity=1.1,
            fill_direction=(-0.7, 0.3, 0.8),
        ),
        render_meshes(
            [
                (object_vertices, object_faces, OBJECT),
                (mano_vertices, mano_faces, MANO_GHOST),
                (robot_vertices, robot_faces, ROBOT),
            ],
            direction,
            size,
            contact_points=contact_points,
            ambient_intensity=0.48,
            key_intensity=2.5,
            fill_intensity=1.1,
            fill_direction=(-0.7, 0.3, 0.8),
        ),
    ]
    position_mm = float(robot_result["position_rmse_m"][row]) * 1000.0
    orientation_deg = float(np.rad2deg(robot_result["orientation_mean_rad"][row]))
    titles = [
        "Morphology + Contact MANO",
        f"Recovered Wuji Hand2\n{position_mm:.2f} mm target RMSE",
        f"Robot / MANO Overlay\n{orientation_deg:.1f}° orientation mean",
    ]
    canvas = Image.new("RGBA", (1430, 550), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        x = 5 + index * 475
        canvas.alpha_composite(panel, (x, 78))
        draw_centered_text(
            draw, (x, 12, x + 470, 75), title, load_font(22, bold=True)
        )
    draw_centered_text(
        draw,
        (0, 515, 1430, 545),
        f"Frame {args.frame}  |  red = MANO contact targets  |  blue = Wuji Hand2",
        load_font(17),
    )
    save_image(canvas.convert("RGB"), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
