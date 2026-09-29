"""Render a paper-style reference/contact-matched comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...utils.io import save_image
from ...visualization.common import (
    BACKGROUND,
    draw_centered_text,
    load_font,
    render_meshes,
)


SKIN = (0.78, 0.55, 0.42, 1.0)
MATCHED = (0.40, 0.66, 0.79, 1.0)
REFERENCE_GHOST = (0.78, 0.55, 0.42, 0.32)
OBJECT = (0.72, 0.78, 0.71, 0.88)


def main() -> None:
    """Render reference, warm-start, optimized, and overlay contact panels."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", type=Path, default=Path("outputs/wuji_hand2/contact_match_full.npz")
    )
    parser.add_argument(
        "--contacts", type=Path, default=Path("../data/processed/s1/apple_lift_contacts.npz")
    )
    parser.add_argument("--frame", type=int, default=1732)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wuji_hand2/contact_matching.png")
    )
    args = parser.parse_args()

    with np.load(args.result, allow_pickle=False) as result:
        result_frames = result["frame_index"]
        matches = np.flatnonzero(result_frames == args.frame)
        if len(matches) != 1:
            raise ValueError(f"frame {args.frame} is not present exactly once in {args.result}")
        row = int(matches[0])
        reference = result["reference_vertices_m"][row]
        warm_start = result["initial_vertices_m"][row]
        optimized = result["optimized_vertices_m"][row]
        hand_faces = result["mano_faces"]
        before_rmse = float(result["initial_contact_rmse_m"][row]) * 1000.0
        after_rmse = float(result["optimized_contact_rmse_m"][row]) * 1000.0

    with np.load(args.contacts, allow_pickle=False) as contacts:
        transform = contacts["object_to_world"][args.frame]
        object_vertices = (
            contacts["object_vertices_local_m"] @ transform[:3, :3].T + transform[:3, 3]
        )
        object_faces = contacts["object_faces"]
        offsets = contacts["contact_offsets"]
        start, stop = int(offsets[args.frame]), int(offsets[args.frame + 1])
        contact_indices = contacts["contact_indices"][start:stop]

    reference_contacts = reference[contact_indices]
    warm_contacts = warm_start[contact_indices]
    optimized_contacts = optimized[contact_indices]
    directions = ((1.0, -1.2, 0.72), (-0.8, -1.1, 0.58))
    cell_size = (390, 390)
    panels = [
        render_meshes(
            [(object_vertices, object_faces, OBJECT), (reference, hand_faces, SKIN)],
            directions[0],
            cell_size,
            contact_points=reference_contacts,
            ambient_intensity=0.48,
            key_intensity=2.5,
            fill_intensity=1.1,
            fill_direction=(-0.7, 0.3, 0.8),
        ),
        render_meshes(
            [(object_vertices, object_faces, OBJECT), (warm_start, hand_faces, MATCHED)],
            directions[0],
            cell_size,
            contact_points=warm_contacts,
            ambient_intensity=0.48,
            key_intensity=2.5,
            fill_intensity=1.1,
            fill_direction=(-0.7, 0.3, 0.8),
        ),
        render_meshes(
            [(object_vertices, object_faces, OBJECT), (optimized, hand_faces, MATCHED)],
            directions[0],
            cell_size,
            contact_points=optimized_contacts,
            ambient_intensity=0.48,
            key_intensity=2.5,
            fill_intensity=1.1,
            fill_direction=(-0.7, 0.3, 0.8),
        ),
        render_meshes(
            [
                (object_vertices, object_faces, OBJECT),
                (reference, hand_faces, REFERENCE_GHOST),
                (optimized, hand_faces, MATCHED),
            ],
            directions[1],
            cell_size,
            contact_points=optimized_contacts,
            ambient_intensity=0.48,
            key_intensity=2.5,
            fill_intensity=1.1,
            fill_direction=(-0.7, 0.3, 0.8),
        ),
    ]

    canvas = Image.new("RGBA", (1600, 510), BACKGROUND)
    for index, panel in enumerate(panels):
        canvas.alpha_composite(panel, (5 + 398 * index, 80))
    draw = ImageDraw.Draw(canvas)
    titles = [
        "Reference Human Contact",
        f"Sequential Warm Start\n{before_rmse:.2f} mm",
        f"Contact Matched MANO\n{after_rmse:.2f} mm",
        "Matched / Reference Overlay",
    ]
    for index, title in enumerate(titles):
        draw_centered_text(
            draw,
            (5 + 398 * index, 14, 395 + 398 * index, 75),
            title,
            load_font(20, bold=True),
        )
    draw.line((792, 95, 792, 465), fill=(170, 174, 177, 255), width=2)
    draw_centered_text(
        draw,
        (0, 470, 1600, 505),
        f"Frame {args.frame}  |  red = transferred contact vertices",
        load_font(18),
    )
    save_image(canvas.convert("RGB"), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
