"""Render Scaled-MANO morphology alignment against the Sharpa URDF meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from ...optimization.alignment import palm_frame
from ...utils.io import load_yaml
from ...visualization.urdf_geometry import load_urdf_visual_geometry


def _surface(
    axis,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: str,
    alpha: float,
    *,
    edgecolor: str = "none",
) -> None:
    """Add one transparent triangle surface to a Matplotlib 3D axis."""

    collection = Poly3DCollection(
        vertices[faces],
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=0.05,
        alpha=alpha,
        rasterized=True,
    )
    axis.add_collection3d(collection)


def _equal_limits(axis, points: np.ndarray) -> None:
    """Apply equal xyz scale around the supplied point cloud."""

    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = float(np.max(upper - lower) * 0.51)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _draw_overlay(
    axis,
    mano_vertices: np.ndarray,
    mano_faces: np.ndarray,
    mano_points: np.ndarray,
    robot_vertices: np.ndarray,
    robot_faces: np.ndarray,
    robot_points: np.ndarray,
    title: str,
    view: tuple[float, float, float],
    show_correspondence: bool = True,
) -> None:
    """Draw one MANO/robot overlay with optional semantic correspondences."""

    _surface(axis, robot_vertices, robot_faces, "#5296b8", 0.31)
    _surface(axis, mano_vertices, mano_faces, "#ef7f4d", 0.58)
    if show_correspondence:
        for human, robot in zip(mano_points, robot_points, strict=True):
            axis.plot(
                [human[0], robot[0]],
                [human[1], robot[1]],
                [human[2], robot[2]],
                color="#35424a",
                linewidth=0.65,
                alpha=0.58,
            )
        axis.scatter(*mano_points.T, s=9, color="#bd4e27", depthshade=False)
        axis.scatter(*robot_points.T, s=9, color="#24647f", depthshade=False)
    _equal_limits(axis, np.concatenate([robot_vertices, mano_vertices]))
    axis.view_init(elev=view[0], azim=view[1], roll=view[2])
    axis.set_title(title, fontsize=12, pad=2)
    axis.set_axis_off()
    axis.set_facecolor("#fbfcfd")


def main() -> None:
    """Render diagnostic morphology before/after and multi-view figures."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sharpa_wave_right.yaml"))
    parser.add_argument(
        "--geometry", type=Path, default=Path("outputs/morphology_match.npz")
    )
    parser.add_argument(
        "--comparison", type=Path, default=Path("outputs/morphology_before_after.png")
    )
    parser.add_argument(
        "--views", type=Path, default=Path("outputs/morphology_aligned_views.png")
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    display_name = config.get("display_name", config["hand"])
    robot_vertices, robot_faces = load_urdf_visual_geometry(Path(config["urdf"]))
    with np.load(args.geometry, allow_pickle=False) as data:
        initial_vertices = data["initial_vertices_m"]
        optimized_vertices = data["optimized_vertices_m"]
        mano_faces = data["mano_faces"]
        joint_indices = data["mano_correspondence_joint_indices"]
        tip_indices = data["mano_tip_vertex_indices"]
        initial_points = np.concatenate(
            [data["initial_joints_m"][joint_indices], initial_vertices[tip_indices]]
        )
        optimized_points = np.concatenate(
            [data["optimized_joints_m"][joint_indices], optimized_vertices[tip_indices]]
        )
        robot_points = np.concatenate(
            [data["robot_joint_targets_m"], data["robot_tip_targets_m"]]
        )
    frame = palm_frame(
        optimized_points[0], optimized_points[16], optimized_points[18]
    )
    lateral, longitudinal, normal = frame.T

    def view_angles(direction: np.ndarray) -> tuple[float, float, float]:
        """Convert a world-space view direction to Matplotlib Euler angles."""

        direction = direction / np.linalg.norm(direction)
        elevation = float(np.degrees(np.arcsin(direction[2])))
        azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])))
        roll = 0.0 if longitudinal[2] >= 0 else 180.0
        return elevation, azimuth, roll

    palm_view = view_angles(normal)
    oblique_view = view_angles(normal - 0.48 * lateral + 0.24 * longitudinal)
    side_view = view_angles(-lateral + 0.08 * normal)
    initial_joint_rmse = np.sqrt(
        np.mean(np.sum((initial_points[:16] - robot_points[:16]) ** 2, axis=1))
    )
    initial_tip_rmse = np.sqrt(
        np.mean(np.sum((initial_points[16:] - robot_points[16:]) ** 2, axis=1))
    )
    optimized_joint_rmse = np.sqrt(
        np.mean(np.sum((optimized_points[:16] - robot_points[:16]) ** 2, axis=1))
    )
    optimized_tip_rmse = np.sqrt(
        np.mean(np.sum((optimized_points[16:] - robot_points[16:]) ** 2, axis=1))
    )

    legend = [
        Line2D([0], [0], marker="s", linestyle="none", color="#ef7f4d", label="Scaled MANO"),
        Line2D([0], [0], marker="s", linestyle="none", color="#5296b8", label=display_name),
        Line2D([0], [0], color="#35424a", linewidth=1, label="Semantic correspondence"),
    ]
    figure = plt.figure(figsize=(13.5, 6.7), dpi=180, facecolor="#fbfcfd")
    before = figure.add_subplot(1, 2, 1, projection="3d")
    after = figure.add_subplot(1, 2, 2, projection="3d")
    view = palm_view
    _draw_overlay(
        before,
        initial_vertices,
        mano_faces,
        initial_points,
        robot_vertices,
        robot_faces,
        robot_points,
        f"Before optimization\nJoints {initial_joint_rmse * 1000:.1f} mm | "
        f"Fingertips {initial_tip_rmse * 1000:.1f} mm",
        view,
    )
    _draw_overlay(
        after,
        optimized_vertices,
        mano_faces,
        optimized_points,
        robot_vertices,
        robot_faces,
        robot_points,
        f"After morphology matching\nJoints {optimized_joint_rmse * 1000:.1f} mm | "
        f"Fingertips {optimized_tip_rmse * 1000:.1f} mm",
        view,
    )
    figure.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=10)
    figure.suptitle(f"MANO to {display_name} morphology matching", fontsize=16, y=0.985)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.10, wspace=0.01)
    args.comparison.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.comparison, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    view_specs = [
        ("Palm", palm_view),
        ("Oblique", oblique_view),
        ("Side", side_view),
    ]
    figure = plt.figure(figsize=(15.5, 5.3), dpi=180, facecolor="#fbfcfd")
    for index, (name, camera_view) in enumerate(view_specs, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        _draw_overlay(
            axis,
            optimized_vertices,
            mano_faces,
            optimized_points,
            robot_vertices,
            robot_faces,
            robot_points,
            name,
            camera_view,
            show_correspondence=False,
        )
    figure.legend(handles=legend[:2], loc="lower center", ncol=2, frameon=False, fontsize=10)
    figure.suptitle(
        f"Morphology-aligned Scaled MANO over {display_name}", fontsize=16, y=0.97
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.10, wspace=0.0)
    args.views.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.views, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    print(args.comparison)
    print(args.views)


if __name__ == "__main__":
    main()
