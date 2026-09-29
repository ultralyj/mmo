"""Policy-rollout visualization for residual-RL diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..utils.io import save_gif


def _rgb_image(rendered: Any) -> Image.Image:
    """Convert a ManiSkill batched render result to an RGB Pillow image."""

    if isinstance(rendered, torch.Tensor):
        array = rendered.detach().cpu().numpy()
    else:
        array = np.asarray(rendered)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] > 3:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _annotate_frame(
    frame: Image.Image, step: int, reward: float, info: dict[str, Any]
) -> Image.Image:
    """Overlay the most useful trajectory and termination diagnostics."""

    image = frame.copy()
    draw = ImageDraw.Draw(image)
    add = float(info["add_error"][0]) if "add_error" in info else 0.0
    contact = (
        int(info["observed_contact"][0].sum().item())
        if "observed_contact" in info
        else 0
    )
    reason_names = {
        "terminate_add_ema": "ADD EMA",
        "terminate_contact_ema": "contact EMA",
        "terminate_table_height": "table height",
        "terminate_table_force": "table force",
        "terminate_contact_force": "object force",
    }
    reasons = [
        label
        for key, label in reason_names.items()
        if key in info and bool(info[key][0].item())
    ]
    lines = [
        f"step {step}   reward {reward:.3f}",
        f"ADD {add:.3f} m   contacting fingers {contact}",
        "termination: " + (", ".join(reasons) if reasons else "none"),
    ]
    draw.rectangle((8, 8, 405, 70), fill=(0, 0, 0))
    draw.multiline_text((14, 12), "\n".join(lines), fill=(255, 255, 255), spacing=3)
    return image


@torch.no_grad()
def render_policy_gif(
    policy: Any,
    environment: Any,
    output_path: str | Path,
    *,
    fps: float = 15.0,
    frame_stride: int = 4,
    seed: int = 1,
) -> None:
    """Run one deterministic episode and save an annotated diagnostic GIF."""

    if frame_stride <= 0:
        raise ValueError("GIF frame stride must be positive")
    observation, info = environment.reset(seed=seed)
    if observation.ndim == 1:
        observation = observation.unsqueeze(0)
    frames = [_annotate_frame(_rgb_image(environment.render()), 0, 0.0, info)]
    maximum_steps = environment.unwrapped.reference.frame_count
    for step in range(maximum_steps):
        distribution, _ = policy.forward(observation)
        action = policy.action_limit * torch.tanh(distribution.mean)
        observation, reward, terminated, truncated, info = environment.step(action)
        done = bool((terminated | truncated)[0].item())
        if step % frame_stride == 0 or done:
            frames.append(
                _annotate_frame(
                    _rgb_image(environment.render()),
                    step + 1,
                    float(reward[0].item()),
                    info,
                )
            )
        if done:
            break
    save_gif(frames, Path(output_path), fps=fps)
