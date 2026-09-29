"""Render a fixed-camera GIF of recovered robot motion and MANO contacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import numpy as np
from PIL import Image, ImageDraw

from ...utils.frames import parse_frame_indices
from ...utils.io import load_yaml, save_gif
from ...visualization.common import (
    BACKGROUND,
    draw_centered_text,
    load_font,
    render_meshes,
)
from ...visualization.urdf_geometry import load_urdf_visual_geometry


ROBOT = (0.37, 0.65, 0.82, 1.0)
OBJECT = (0.72, 0.78, 0.71, 0.90)


def _result_rows(frame_index: np.ndarray) -> dict[int, int]:
    """Map unique source-frame indices to rows in a result archive."""

    values = np.asarray(frame_index, dtype=np.int64)
    if len(np.unique(values)) != len(values):
        raise ValueError("result frame_index values must be unique")
    return {int(frame): row for row, frame in enumerate(values)}


def main() -> None:
    """Render selected recovered-pose frames as a looping contact GIF."""

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
    parser.add_argument(
        "--frames",
        default="1450:1951:8",
        help="all, comma-separated indices, or START:STOP[:STRIDE]",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--world-frame",
        action="store_true",
        help="Keep world coordinates instead of stabilizing the manipulated object.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wuji_hand2/contact_recovery.gif")
    )
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    config = load_yaml(args.config)
    robot_result = np.load(args.robot_pose, allow_pickle=False)
    contact_result = np.load(args.contact_match, allow_pickle=False)
    contacts = np.load(args.contacts, allow_pickle=False)
    total_frames = len(contacts["timestamp_s"])
    selected = parse_frame_indices(args.frames, total_frames)
    robot_rows = _result_rows(robot_result["frame_index"])
    contact_rows = _result_rows(contact_result["frame_index"])
    unavailable = [
        int(frame)
        for frame in selected
        if int(frame) not in robot_rows or int(frame) not in contact_rows
    ]
    if unavailable:
        raise ValueError(
            f"{len(unavailable)} selected frames are absent from pose/contact results"
        )

    joint_names = [str(value) for value in robot_result["joint_names"]]
    object_faces = contacts["object_faces"]
    offsets = contacts["contact_offsets"]
    prepared = []
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for output_index, frame_value in enumerate(selected):
        frame = int(frame_value)
        robot_row = robot_rows[frame]
        contact_row = contact_rows[frame]
        configuration = dict(
            zip(
                joint_names,
                robot_result["joint_positions_rad"][robot_row],
                strict=True,
            )
        )
        base_transform = np.asarray(
            jaxlie.SE3(
                jnp.asarray(robot_result["base_wxyz_xyz"][robot_row])
            ).as_matrix()
        )
        robot_vertices, robot_faces = load_urdf_visual_geometry(
            Path(config["urdf"]), configuration, base_transform
        )
        object_transform = contacts["object_to_world"][frame]
        object_vertices = (
            contacts["object_vertices_local_m"] @ object_transform[:3, :3].T
            + object_transform[:3, 3]
        )
        mano_vertices = contact_result["optimized_vertices_m"][contact_row]
        start, stop = int(offsets[frame]), int(offsets[frame + 1])
        contact_indices = contacts["contact_indices"][start:stop]
        contact_points = mano_vertices[contact_indices]
        if not args.world_frame:
            rotation = object_transform[:3, :3]
            translation = object_transform[:3, 3]
            robot_vertices = (robot_vertices - translation) @ rotation
            object_vertices = (object_vertices - translation) @ rotation
            contact_points = (contact_points - translation) @ rotation
        geometry = np.concatenate([robot_vertices, object_vertices], axis=0)
        lower = np.minimum(lower, geometry.min(axis=0))
        upper = np.maximum(upper, geometry.max(axis=0))
        prepared.append(
            (
                frame,
                robot_row,
                robot_vertices,
                robot_faces,
                object_vertices,
                contact_points,
            )
        )
        if output_index == 0 or (output_index + 1) % 10 == 0:
            print(f"prepared {output_index + 1}/{len(selected)} frames", flush=True)

    # Add stable padding once so the camera does not breathe between frames.
    padding = 0.06 * float(np.max(upper - lower))
    bounds = (lower - padding, upper + padding)
    frames = []
    detected_counts = contacts["detected_contact_counts"]
    for output_index, item in enumerate(prepared):
        (
            frame,
            robot_row,
            robot_vertices,
            robot_faces,
            object_vertices,
            contact_points,
        ) = item
        panel = render_meshes(
            [
                (object_vertices, object_faces, OBJECT),
                (robot_vertices, robot_faces, ROBOT),
            ],
            (1.0, -1.2, 0.72),
            (640, 590),
            contact_points=contact_points,
            ambient_intensity=0.48,
            key_intensity=2.5,
            fill_intensity=1.1,
            fill_direction=(-0.7, 0.3, 0.8),
            bounds=bounds,
        )
        canvas = Image.new("RGBA", (700, 680), BACKGROUND)
        canvas.alpha_composite(panel, (30, 52))
        draw = ImageDraw.Draw(canvas)
        draw_centered_text(
            draw,
            (0, 10, 700, 50),
            "Wuji Hand2 Contact-Matched Pose Recovery",
            load_font(23, bold=True),
        )
        position_mm = float(robot_result["position_rmse_m"][robot_row]) * 1000.0
        draw_centered_text(
            draw,
            (0, 642, 700, 675),
            f"frame {frame}  |  contacts {int(detected_counts[frame])}  |  "
            f"target RMSE {position_mm:.2f} mm  |  red = MANO contact",
            load_font(15),
        )
        frames.append(canvas.convert("RGB"))
        if output_index == 0 or (output_index + 1) % 10 == 0:
            print(f"rendered {output_index + 1}/{len(selected)} frames", flush=True)

    save_gif(frames, args.output, fps=args.fps)
    print(args.output)


if __name__ == "__main__":
    main()
