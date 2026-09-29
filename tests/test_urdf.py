from pathlib import Path

import yaml

from morphim_mmo.models.urdf import UrdfModel, extract_semantic_points


PROJECT = Path(__file__).resolve().parents[1]


def test_sharpa_urdf_has_expected_dof_and_semantic_points() -> None:
    config = yaml.safe_load((PROJECT / "configs/sharpa_wave_right.yaml").read_text())
    model = UrdfModel.from_file(PROJECT / config["urdf"])
    semantics = config["robot_semantics"]
    wrist, mcps, tips = extract_semantic_points(
        model,
        config["anatomical_wrist_link"],
        {name: values["mcp_joint"] for name, values in semantics.items()},
        {name: values["tip_joint"] for name, values in semantics.items()},
    )
    assert len(model.actuated_joints) == 22
    assert wrist.shape == (3,)
    assert set(mcps) == set(config["fingers"])
    assert set(tips) == set(config["fingers"])
    for name in config["fingers"]:
        assert ((tips[name] - mcps[name]) ** 2).sum() > 0.0


def test_wuji_hand2_beta2_has_expected_dof_and_semantic_points() -> None:
    config = yaml.safe_load((PROJECT / "configs/wuji_hand2_beta2_right.yaml").read_text())
    model = UrdfModel.from_file(PROJECT / config["urdf"])
    semantics = config["robot_semantics"]
    wrist, mcps, tips = extract_semantic_points(
        model,
        config["anatomical_wrist_link"],
        {name: values["mcp_joint"] for name, values in semantics.items()},
        {name: values["tip_joint"] for name, values in semantics.items()},
    )
    assert model.root == "r_wrist"
    assert len(model.actuated_joints) == 20
    assert wrist.shape == (3,)
    assert set(mcps) == set(config["fingers"])
    assert set(tips) == set(config["fingers"])
    for name in config["fingers"]:
        assert ((tips[name] - mcps[name]) ** 2).sum() > 0.0
