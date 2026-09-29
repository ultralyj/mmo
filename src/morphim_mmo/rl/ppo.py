"""Minimal PyTorch PPO for bounded residual joint actions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.distributions import Normal

from .config import PPOConfig


_TERMINATION_REASON_KEYS = (
    "terminate_add_ema",
    "terminate_contact_ema",
    "terminate_table_height",
    "terminate_table_force",
    "terminate_contact_force",
)


def _orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)


def _single_env_action_limit(
    high: Any, action_dim: int, device: torch.device
) -> torch.Tensor:
    """Extract one action bound vector from a possibly batched Gym space."""

    limits = torch.as_tensor(high, dtype=torch.float32, device=device)
    if limits.numel() % action_dim:
        raise ValueError(
            f"action bounds with shape {tuple(limits.shape)} are incompatible "
            f"with action dimension {action_dim}"
        )
    limits = limits.reshape(-1, action_dim)
    if not torch.allclose(limits, limits[:1].expand_as(limits)):
        raise ValueError("all vectorized environments must use equal action bounds")
    if not torch.all(torch.isfinite(limits[0]) & (limits[0] > 0)):
        raise ValueError("residual action limits must be finite and positive")
    return limits[0]


class ActorCritic(nn.Module):
    """A Gaussian residual policy and a state-value critic."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
        action_limit: torch.Tensor,
    ) -> None:
        super().__init__()
        policy_layers: list[nn.Module] = []
        previous = observation_dim
        for hidden in hidden_sizes:
            layer = nn.Linear(previous, hidden)
            _orthogonal(layer, gain=nn.init.calculate_gain("relu"))
            policy_layers.extend((layer, nn.ReLU()))
            previous = hidden
        self.policy_body = nn.Sequential(*policy_layers)
        self.policy_mean = nn.Linear(previous, action_dim)
        _orthogonal(self.policy_mean, gain=0.01)
        self.policy_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.register_buffer("action_limit", action_limit.clone())

        value_layers: list[nn.Module] = []
        previous = observation_dim
        for hidden in hidden_sizes:
            layer = nn.Linear(previous, hidden)
            _orthogonal(layer, gain=nn.init.calculate_gain("relu"))
            value_layers.extend((layer, nn.ReLU()))
            previous = hidden
        self.value_body = nn.Sequential(*value_layers)
        self.value_head = nn.Linear(previous, 1)
        _orthogonal(self.value_head, gain=1.0)

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[Normal, torch.Tensor]:
        policy = Normal(
            self.policy_mean(self.policy_body(observation)),
            self.policy_log_std.exp().expand(observation.shape[0], -1),
        )
        value = self.value_head(self.value_body(observation)).squeeze(-1)
        return policy, value

    def act(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy, value = self.forward(observation)
        raw_action = policy.sample()
        squashed = torch.tanh(raw_action)
        action = self.action_limit * squashed
        log_probability = (
            policy.log_prob(raw_action)
            - torch.log(self.action_limit * (1.0 - squashed.square()) + 1e-6)
        ).sum(dim=-1)
        entropy = -log_probability
        return action, log_probability, entropy, value

    def evaluate(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy, value = self.forward(observation)
        squashed = (action / self.action_limit).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = torch.atanh(squashed)
        log_probability = (
            policy.log_prob(raw_action)
            - torch.log(self.action_limit * (1.0 - squashed.square()) + 1e-6)
        ).sum(dim=-1)
        entropy = -log_probability
        return log_probability, entropy, value


class _Rollout:
    """Fixed-size vectorized PPO rollout storage."""

    def __init__(
        self,
        steps: int,
        num_envs: int,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        shape = (steps, num_envs)
        self.observations = torch.zeros(
            (*shape, observation_dim), device=device
        )
        self.actions = torch.zeros((*shape, action_dim), device=device)
        self.log_probabilities = torch.zeros(shape, device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)
        self.next_values = torch.zeros(shape, device=device)
        self.done = torch.zeros(
            shape, dtype=torch.bool, device=device
        )


class PPOTrainer:
    """Train a residual teacher policy with clipped-surrogate PPO."""

    def __init__(self, env: Any, config: PPOConfig) -> None:
        self.env = env
        self.config = config
        self.base_env = env.unwrapped
        self.device = self.base_env.device
        torch.manual_seed(config.seed)

        observation_dim = int(env.observation_space.shape[-1])
        action_dim = int(env.action_space.shape[-1])
        action_limit = _single_env_action_limit(
            env.action_space.high, action_dim, self.device
        )
        self.policy = ActorCritic(
            observation_dim,
            action_dim,
            config.hidden_sizes,
            action_limit,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )

    def _advantages_and_returns(
        self, rollout: _Rollout
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rollout.rewards)
        gae = torch.zeros(self.config.num_envs, device=self.device)
        for step in reversed(range(self.config.rollout_steps)):
            non_terminal = ~rollout.done[step]
            delta = (
                rollout.rewards[step]
                + self.config.gamma
                * rollout.next_values[step]
                * non_terminal
                - rollout.values[step]
            )
            gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * non_terminal
                * gae
            )
            advantages[step] = gae
        returns = advantages + rollout.values
        return advantages, returns

    def _update(self, rollout: _Rollout) -> dict[str, float]:
        advantages, returns = self._advantages_and_returns(rollout)
        observations = rollout.observations.flatten(end_dim=1)
        actions = rollout.actions.flatten(end_dim=1)
        old_log_probabilities = rollout.log_probabilities.flatten()
        advantages = advantages.flatten()
        returns = returns.flatten()
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False).clamp_min(1e-8)
        )

        batch_size = observations.shape[0]
        minibatch_size = max(
            1, batch_size // self.config.minibatches
        )
        policy_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(
                batch_size, device=self.device
            )
            for start in range(
                0, batch_size, minibatch_size
            ):
                indices = permutation[
                    start : start + minibatch_size
                ]
                log_probability, entropy, value = (
                    self.policy.evaluate(
                        observations[indices],
                        actions[indices],
                    )
                )
                ratio = (
                    log_probability
                    - old_log_probabilities[indices]
                ).exp()
                surrogate = torch.minimum(
                    ratio * advantages[indices],
                    ratio.clamp(
                        1.0 - self.config.clip_coefficient,
                        1.0 + self.config.clip_coefficient,
                    )
                    * advantages[indices],
                )
                policy_loss = -surrogate.mean()
                value_loss = torch.nn.functional.mse_loss(
                    value, returns[indices]
                )
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient
                    * value_loss
                    + self.config.entropy_coefficient
                    * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_gradient_norm,
                )
                self.optimizer.step()
                policy_losses.append(
                    policy_loss.detach()
                )
                value_losses.append(
                    value_loss.detach()
                )
                entropies.append(entropy.detach().mean())

        return {
            "policy_loss": float(
                torch.stack(policy_losses).mean()
            ),
            "value_loss": float(
                torch.stack(value_losses).mean()
            ),
            "entropy": float(
                torch.stack(entropies).mean()
            ),
        }

    def _reset_finished(
        self,
        observation: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        indices = torch.nonzero(
            done, as_tuple=False
        ).flatten()
        if indices.numel() == 0:
            return observation
        reset_observation, _ = self.env.reset(
            options={"env_idx": indices}
        )
        return reset_observation

    def train(
        self,
        *,
        output_dir: str | Path,
        checkpoint_interval: int = 10,
        visualization_interval: int = 0,
        visualization_callback: Callable[[ActorCritic, int, Path], None]
        | None = None,
    ) -> list[dict[str, float]]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        existing = tuple(output.iterdir())
        run_artifacts = tuple(path for path in existing if path.name != "config.json")
        if run_artifacts:
            raise FileExistsError(
                f"training output directory is not empty: {output}; "
                "choose a new directory to avoid mixing runs"
            )
        (output / "config.json").write_text(
            json.dumps(
                asdict(self.config), indent=2
            ),
            encoding="utf-8",
        )

        observation, _ = self.env.reset(
            seed=self.config.seed
        )
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)

        rollout = _Rollout(
            self.config.rollout_steps,
            self.config.num_envs,
            int(observation.shape[-1]),
            int(self.env.action_space.shape[-1]),
            self.device,
        )

        history: list[dict[str, float]] = []
        collected = 0
        iteration = 0

        start_time = time.perf_counter()
        steps_per_iteration = (
            self.config.rollout_steps
            * self.config.num_envs
        )
        rollout_progress_interval = max(
            1, self.config.rollout_steps // 10
        )

        print(
            "Starting PPO training | "
            f"device={self.device} | "
            f"num_envs={self.config.num_envs} | "
            f"rollout_steps={self.config.rollout_steps} | "
            f"steps/iter={steps_per_iteration:,} | "
            f"total_timesteps="
            f"{self.config.total_timesteps:,}",
            flush=True,
        )

        while (
            collected
            < self.config.total_timesteps
        ):
            iteration_start = time.perf_counter()

            print(
                f"\n[Iteration {iteration + 1}] "
                "Collecting rollout...",
                flush=True,
            )
            completed_episodes = 0
            successful_episodes = 0
            reference_end_episodes = 0
            termination_counts = {key: 0 for key in _TERMINATION_REASON_KEYS}
            diagnostic_maxima = {
                "max_add_ema_m": 0.0,
                "max_contact_mismatch_ema": 0.0,
                "max_object_contact_force_n": 0.0,
                "max_table_contact_force_n": 0.0,
            }
            minimum_table_clearance = float("inf")

            for step in range(
                self.config.rollout_steps
            ):
                rollout.observations[step] = (
                    observation
                )
                with torch.no_grad():
                    (
                        action,
                        log_probability,
                        _,
                        value,
                    ) = self.policy.act(
                        observation
                    )
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.env.step(action)

                if next_observation.ndim == 1:
                    next_observation = (
                        next_observation.unsqueeze(0)
                    )
                with torch.no_grad():
                    _, next_value = self.policy.forward(next_observation)

                done = terminated | truncated
                completed_episodes += int(done.sum().item())
                reference_end_episodes += int(
                    (truncated & ~terminated).sum().item()
                )
                if "success" in info:
                    successful_episodes += int((info["success"] & done).sum().item())
                for key in _TERMINATION_REASON_KEYS:
                    if key in info:
                        termination_counts[key] += int(
                            (info[key] & terminated).sum().item()
                        )
                diagnostic_sources = {
                    "max_add_ema_m": "add_ema",
                    "max_contact_mismatch_ema": "contact_mismatch_ema",
                    "max_object_contact_force_n": (
                        "maximum_object_contact_force"
                    ),
                    "max_table_contact_force_n": "table_contact_force",
                }
                for metric_name, info_name in diagnostic_sources.items():
                    if info_name in info:
                        diagnostic_maxima[metric_name] = max(
                            diagnostic_maxima[metric_name],
                            float(info[info_name].max().item()),
                        )
                if "table_clearance" in info:
                    minimum_table_clearance = min(
                        minimum_table_clearance,
                        float(info["table_clearance"].min().item()),
                    )

                rollout.actions[step] = action
                rollout.log_probabilities[
                    step
                ] = log_probability
                rollout.rewards[step] = reward
                rollout.values[step] = value
                rollout.next_values[
                    step
                ] = next_value
                rollout.done[
                    step
                ] = done

                collected += (
                    self.config.num_envs
                )
                observation = (
                    self._reset_finished(
                        next_observation,
                        done,
                    )
                )

                if (
                    (step + 1)
                    % rollout_progress_interval
                    == 0
                    or step + 1
                    == self.config.rollout_steps
                ):
                    rollout_percent = (
                        100.0
                        * (step + 1)
                        / self.config.rollout_steps
                    )
                    done_count = int(
                        (
                            terminated
                            | truncated
                        )
                        .sum()
                        .item()
                    )
                    # print(
                    #     "  rollout "
                    #     f"{step + 1:>4}/"
                    #     f"{self.config.rollout_steps} "
                    #     f"({rollout_percent:5.1f}%) | "
                    #     f"total_steps="
                    #     f"{collected:,} | "
                    #     f"done={done_count}",
                    #     flush=True,
                    # )

            # print(
            #     "  Updating PPO...",
            #     flush=True,
            # )
            metrics = self._update(rollout)

            metrics["iteration"] = iteration
            metrics["timesteps"] = float(
                collected
            )
            metrics["mean_reward"] = float(
                rollout.rewards.mean()
            )
            metrics["completed_episodes"] = float(completed_episodes)
            metrics["successful_episodes"] = float(successful_episodes)
            metrics["reference_end_episodes"] = float(reference_end_episodes)
            for key, count in termination_counts.items():
                metrics[f"{key}_count"] = float(count)
            metrics.update(diagnostic_maxima)
            metrics["min_table_clearance_m"] = (
                minimum_table_clearance
                if minimum_table_clearance != float("inf")
                else 0.0
            )
            metrics["success_rate"] = (
                successful_episodes / completed_episodes
                if completed_episodes
                else 0.0
            )

            iteration_time = (
                time.perf_counter()
                - iteration_start
            )
            elapsed = (
                time.perf_counter()
                - start_time
            )
            progress = min(
                100.0,
                100.0
                * collected
                / self.config.total_timesteps,
            )
            fps = collected / max(
                elapsed, 1e-8
            )
            remaining_steps = max(
                0,
                self.config.total_timesteps
                - collected,
            )
            eta_seconds = (
                remaining_steps
                / max(fps, 1e-8)
            )

            print(
                f"[Iteration {iteration + 1}] "
                f"progress={progress:6.2f}% | "
                f"steps={collected:,}/"
                f"{self.config.total_timesteps:,} | "
                f"reward="
                f"{metrics['mean_reward']:.4f} | "
                f"success="
                f"{metrics['success_rate']:.3f} | "
                f"policy_loss="
                f"{metrics['policy_loss']:.4f} | "
                f"value_loss="
                f"{metrics['value_loss']:.4f} | "
                f"entropy="
                f"{metrics['entropy']:.4f} | "
                f"time={iteration_time:.1f}s | "
                f"FPS={fps:.0f} | "
                f"ETA={eta_seconds / 60.0:.1f} min",
                flush=True,
            )
            print(
                "  episode outcomes | "
                f"completed={completed_episodes} | "
                f"reference_end={reference_end_episodes} | "
                f"success={successful_episodes} | "
                + " | ".join(
                    f"{key.removeprefix('terminate_')}={count}"
                    for key, count in termination_counts.items()
                ),
                flush=True,
            )

            history.append(metrics)
            (output / "metrics.json").write_text(
                json.dumps(
                    history, indent=2
                ),
                encoding="utf-8",
            )

            iteration += 1
            if (
                visualization_callback is not None
                and visualization_interval > 0
                and iteration % visualization_interval == 0
            ):
                gif_path = output / "gifs" / f"policy-{iteration:06d}.gif"
                visualization_callback(self.policy, iteration, gif_path)
                print(f"  Saved policy GIF: {gif_path}", flush=True)
            if (
                iteration
                % checkpoint_interval
                == 0
            ):
                checkpoint_path = (
                    output
                    / (
                        f"checkpoint-"
                        f"{iteration:06d}.pt"
                    )
                )
                torch.save(
                    {
                        "policy": self.policy.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "iteration": iteration,
                        "timesteps": collected,
                        "config": asdict(self.config),
                    },
                    checkpoint_path,
                )
                print(
                    "  Saved checkpoint: "
                    f"{checkpoint_path}",
                    flush=True,
                )

        final_checkpoint = (
            output / "checkpoint-final.pt"
        )
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": iteration,
                "timesteps": collected,
                "config": asdict(self.config),
            },
            final_checkpoint,
        )
        print(
            "Training complete. "
            "Saved final checkpoint: "
            f"{final_checkpoint}",
            flush=True,
        )
        return history
