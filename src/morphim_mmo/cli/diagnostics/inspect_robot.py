"""Inspect the pinned Sharpa URDF and emit its neutral semantic skeleton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...models.urdf import UrdfModel, extract_semantic_points
from ...utils.io import load_yaml, write_json


def main() -> None:
    """Inspect a robot URDF and export its neutral semantic skeleton."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sharpa_wave_right.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sharpa_neutral_skeleton.json"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    model = UrdfModel.from_file(config["urdf"])
    semantics = config["robot_semantics"]
    wrist, mcps, tips = extract_semantic_points(
        model,
        config["anatomical_wrist_link"],
        {name: values["mcp_joint"] for name, values in semantics.items()},
        {name: values["tip_joint"] for name, values in semantics.items()},
    )
    payload = {
        "hand": config["hand"],
        "source_revision": config["source_revision"],
        "urdf": config["urdf"],
        "root_link": model.root,
        "actuated_dof": len(model.actuated_joints),
        "actuated_joint_names": [joint.name for joint in model.actuated_joints],
        "anatomical_wrist": wrist.tolist(),
        "mcp_positions": {name: value.tolist() for name, value in mcps.items()},
        "fingertip_positions": {name: value.tolist() for name, value in tips.items()},
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "actuated_dof": payload["actuated_dof"]}))


if __name__ == "__main__":
    main()
