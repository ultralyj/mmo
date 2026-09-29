"""Configuration objects for the residual-RL reproduction."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class RewardConfig:
    """Paper reward parameters and success thresholds."""

    add_alpha: float
    final_add_success_m: float
    contact_force_threshold_n: float


@dataclass(frozen=True)
class TerminationConfig:
    """EMA and immediate safety termination thresholds."""

    ema_decay: float
    add_ema_threshold_m: float
    contact_mismatch_ema_threshold: float
    maximum_contact_force_n: float
    table_collision_force_n: float


@dataclass(frozen=True)
class ControlConfig:
    """PD controller and residual-action settings."""

    simulation_frequency_hz: int
    control_frequency_hz: int
    reference_stride: int
    future_steps: int
    residual_limit_rad: float
    stiffness: float
    damping: float
    force_limit: float


@dataclass(frozen=True)
class PhysicsConfig:
    """Initial fixed physics parameters for the first debugging stage."""

    object_mass_kg: float
    object_friction: float
    hand_friction: float
    table_friction: float


@dataclass(frozen=True)
class RandomizationConfig:
    """Initial domain-randomization settings from paper Section V."""

    enabled: bool
    object_position_range_m: float
    object_yaw_range_rad: float
    random_start: bool
    external_force_range_n: float


@dataclass(frozen=True)
class PPOConfig:
    """PPO reproduction assumptions absent from the available paper text."""

    seed: int
    num_envs: int
    rollout_steps: int
    total_timesteps: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coefficient: float
    entropy_coefficient: float
    value_coefficient: float
    max_gradient_norm: float
    update_epochs: int
    minibatches: int
    hidden_sizes: tuple[int, ...]


@dataclass(frozen=True)
class ResidualRLConfig:
    """Complete configuration for one object-category residual teacher."""

    reward: RewardConfig
    termination: TerminationConfig
    control: ControlConfig
    physics: PhysicsConfig
    randomization: RandomizationConfig
    ppo: PPOConfig


ConfigType = TypeVar("ConfigType")


def _construct_dataclass(cls: type[ConfigType], values: dict[str, Any]) -> ConfigType:
    """Construct a strict dataclass and reject misspelled configuration keys."""

    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_residual_rl_config(path: str | Path) -> ResidualRLConfig:
    """Load the residual-RL YAML configuration with strict section schemas."""

    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    expected = {
        "reward",
        "termination",
        "control",
        "physics",
        "randomization",
        "ppo",
    }
    unknown = set(values) - expected
    missing = expected - set(values)
    if unknown or missing:
        raise ValueError(
            f"invalid ResidualRLConfig sections; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    ppo_values = dict(values["ppo"])
    ppo_values["hidden_sizes"] = tuple(ppo_values["hidden_sizes"])
    return ResidualRLConfig(
        reward=_construct_dataclass(RewardConfig, values["reward"]),
        termination=_construct_dataclass(
            TerminationConfig, values["termination"]
        ),
        control=_construct_dataclass(ControlConfig, values["control"]),
        physics=_construct_dataclass(PhysicsConfig, values["physics"]),
        randomization=_construct_dataclass(
            RandomizationConfig, values["randomization"]
        ),
        ppo=_construct_dataclass(PPOConfig, ppo_values),
    )
