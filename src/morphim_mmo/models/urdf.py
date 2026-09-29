"""Small dependency-free URDF kinematics reader used for Sharpa geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class Joint:
    """Minimal URDF joint representation used by forward kinematics."""

    name: str
    kind: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def _vector(element: ET.Element | None, attribute: str, default: str) -> np.ndarray:
    """Read a whitespace-separated numeric XML attribute."""

    text = default if element is None else element.attrib.get(attribute, default)
    return np.fromstring(text, sep=" ", dtype=float)


def _rotation_x(angle: float) -> np.ndarray:
    """Return a right-handed x-axis rotation matrix."""

    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rotation_y(angle: float) -> np.ndarray:
    """Return a right-handed y-axis rotation matrix."""

    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rotation_z(angle: float) -> np.ndarray:
    """Return a right-handed z-axis rotation matrix."""

    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert URDF fixed-axis roll-pitch-yaw angles to a matrix."""

    roll, pitch, yaw = rpy
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert one axis-angle pair to a rotation matrix."""

    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3)
    x, y, z = axis / norm
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assemble a homogeneous transform from rotation and translation."""

    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


class UrdfModel:
    """Small dependency-free URDF tree with neutral/configured kinematics."""

    def __init__(self, links: tuple[str, ...], joints: tuple[Joint, ...]) -> None:
        """Validate the tree root and index its joints by name."""

        self.links = links
        self.joints = joints
        children = {joint.child for joint in joints}
        roots = [link for link in links if link not in children]
        if len(roots) != 1:
            raise ValueError(f"expected one URDF root link, found {roots}")
        self.root = roots[0]
        self.joint_by_name = {joint.name: joint for joint in joints}

    @classmethod
    def from_file(cls, path: str | Path) -> "UrdfModel":
        """Parse links, joints, origins, axes, and limits from a URDF file."""

        root = ET.parse(path).getroot()
        links = tuple(element.attrib["name"] for element in root.findall("link"))
        joints: list[Joint] = []
        for element in root.findall("joint"):
            origin = element.find("origin")
            limit = element.find("limit")
            joints.append(
                Joint(
                    name=element.attrib["name"],
                    kind=element.attrib["type"],
                    parent=element.find("parent").attrib["link"],  # type: ignore[union-attr]
                    child=element.find("child").attrib["link"],  # type: ignore[union-attr]
                    origin_xyz=_vector(origin, "xyz", "0 0 0"),
                    origin_rpy=_vector(origin, "rpy", "0 0 0"),
                    axis=_vector(element.find("axis"), "xyz", "0 0 0"),
                    lower=None if limit is None or "lower" not in limit.attrib else float(limit.attrib["lower"]),
                    upper=None if limit is None or "upper" not in limit.attrib else float(limit.attrib["upper"]),
                )
            )
        return cls(links=links, joints=tuple(joints))

    @property
    def actuated_joints(self) -> tuple[Joint, ...]:
        """Return all non-fixed joints in file order."""

        return tuple(joint for joint in self.joints if joint.kind != "fixed")

    def forward_kinematics(
        self, configuration: dict[str, float] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return root-frame link transforms and joint-origin positions."""

        configuration = {} if configuration is None else configuration
        unknown = set(configuration) - set(self.joint_by_name)
        if unknown:
            raise KeyError(f"unknown joints: {sorted(unknown)}")

        link_transforms = {self.root: np.eye(4)}
        joint_positions: dict[str, np.ndarray] = {}
        remaining = list(self.joints)
        while remaining:
            progressed = False
            for joint in remaining.copy():
                if joint.parent not in link_transforms:
                    continue
                origin = _transform(_rpy_matrix(joint.origin_rpy), joint.origin_xyz)
                joint_frame = link_transforms[joint.parent] @ origin
                joint_positions[joint.name] = joint_frame[:3, 3].copy()
                motion = np.eye(4)
                value = float(configuration.get(joint.name, 0.0))
                if joint.kind in {"revolute", "continuous"}:
                    motion[:3, :3] = _axis_angle_matrix(joint.axis, value)
                elif joint.kind == "prismatic":
                    motion[:3, 3] = joint.axis * value
                elif joint.kind != "fixed":
                    raise NotImplementedError(f"joint type {joint.kind!r} is unsupported")
                link_transforms[joint.child] = joint_frame @ motion
                remaining.remove(joint)
                progressed = True
            if not progressed:
                unresolved = [joint.name for joint in remaining]
                raise ValueError(f"URDF graph is disconnected or cyclic: {unresolved}")
        return link_transforms, joint_positions


def extract_semantic_points(
    model: UrdfModel,
    wrist_link: str,
    mcp_joints: dict[str, str],
    tip_joints: dict[str, str],
    configuration: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return configured wrist, named MCP, and named fingertip positions."""

    link_transforms, joint_positions = model.forward_kinematics(configuration)
    if wrist_link not in link_transforms:
        raise KeyError(f"wrist link {wrist_link!r} not found")
    wrist = link_transforms[wrist_link][:3, 3].copy()
    mcps = {name: joint_positions[joint_name].copy() for name, joint_name in mcp_joints.items()}
    tips = {name: joint_positions[joint_name].copy() for name, joint_name in tip_joints.items()}
    return wrist, mcps, tips
