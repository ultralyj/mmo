"""Load posed URDF visual meshes into a single triangle mesh."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from ..models.urdf import UrdfModel, _rpy_matrix, _transform


def load_urdf_visual_geometry(
    urdf_path: Path,
    configuration: dict[str, float] | None = None,
    root_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return combined visual vertices/faces at a joint and root configuration."""

    model = UrdfModel.from_file(urdf_path)
    link_transforms, _ = model.forward_kinematics(configuration)
    root_transform = np.eye(4) if root_transform is None else root_transform
    root = ET.parse(urdf_path).getroot()
    meshes = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        for visual in link.findall("visual"):
            mesh_element = visual.find("geometry/mesh")
            if mesh_element is None:
                continue
            filename = mesh_element.attrib["filename"]
            if filename.startswith("package://"):
                mesh_path = urdf_path.parent / "meshes" / Path(filename).name
            else:
                mesh_path = (urdf_path.parent / filename).resolve()
            mesh = trimesh.load_mesh(mesh_path, process=False)
            if "scale" in mesh_element.attrib:
                mesh.apply_scale(np.fromstring(mesh_element.attrib["scale"], sep=" "))
            origin = visual.find("origin")
            xyz = (
                np.zeros(3)
                if origin is None
                else np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            )
            rpy = (
                np.zeros(3)
                if origin is None
                else np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            )
            transform = (
                root_transform
                @ link_transforms[link_name]
                @ _transform(_rpy_matrix(rpy), xyz)
            )
            mesh.apply_transform(transform)
            meshes.append(mesh)
    if not meshes:
        raise ValueError(f"URDF contains no visual meshes: {urdf_path}")
    combined = trimesh.util.concatenate(meshes)
    return np.asarray(combined.vertices), np.asarray(combined.faces, dtype=np.int32)

