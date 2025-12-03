import argparse
import os
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from examples.aloha_sim.env import AlohaSimEnvironment
from openpi.policies.aloha_policy import AlohaInputs, AlohaOutputs
from openpi.models import model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.config import get_config

# Default to headless rendering if not set (required for dm_control/mujoco)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def convert_env_obs(gym_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert env obs to Aloha policy input dict (HWC uint8) with fallbacks."""
    # Accept multiple layouts: {"pixels": {...}} or {"images": {...}} etc.
    if "pixels" in gym_obs:
        pixels = gym_obs["pixels"]
    elif "images" in gym_obs:
        pixels = gym_obs["images"]
    else:
        raise KeyError(f"Expected 'pixels' or 'images' in obs, got keys: {list(gym_obs.keys())}")

    state = None
    for key in ("agent_pos", "state", "qpos"):
        if key in gym_obs and gym_obs[key] is not None:
            state = gym_obs[key]
            break
    if state is None:
        raise KeyError(f"Expected 'agent_pos'/'state'/'qpos' in obs, got keys: {list(gym_obs.keys())}")

    # Safely pick a base camera
    if "top" in pixels and pixels["top"] is not None:
        top = pixels["top"]
    elif "cam_high" in pixels and pixels["cam_high"] is not None:
        top = pixels["cam_high"]
    else:
        # fallback to first available
        top = next(iter(pixels.values()))

    return {
        "state": np.asarray(state, dtype=np.float32),
        "images": {
            "cam_high": np.asarray(top),  # [H,W,C] uint8
            "cam_left_wrist": np.asarray(pixels.get("left_wrist", top)),
            "cam_right_wrist": np.asarray(pixels.get("right_wrist", top)),
        },
        "prompt": "Transfer cube",  # default prompt used in training
    }


def make_policy(checkpoint_path: str, config_name: str):
    """Load model + params; return a function obs->actions using sample_actions (BC policy, no RL update here)."""
    cfg = get_config(config_name or "pi0_aloha_sim_tiny_b4")
    # Restore params (pure dict), then use cfg.model.load to build model with weights.
    params_pure = _model.restore_params(checkpoint_path, restore_type=np.ndarray)
    model = cfg.model.load(params_pure)
    model.eval()

    inputs_tf = AlohaInputs(adapt_to_pi=True)
    outputs_tf = AlohaOutputs(adapt_to_pi=True)
    tokenizer = PaligemmaTokenizer(cfg.model.max_token_len)

    def tokenize_prompt(prompt: str, batch: int) -> tuple[np.ndarray, np.ndarray]:
        tokens, mask = tokenizer.tokenize(prompt)
        tokens = jnp.asarray(tokens, dtype=jnp.int32)[None, ...].repeat(batch, axis=0)
        mask = jnp.asarray(mask, dtype=bool)[None, ...].repeat(batch, axis=0)
        return tokens, mask

    def policy_fn(obs_np: Dict[str, Any]) -> np.ndarray:
        data_in = inputs_tf(obs_np)  # maps to dict with image/state keys expected by model transforms
        batch = 1
        tok, tok_mask = tokenize_prompt(obs_np.get("prompt", ""), batch)

        # Ensure images have batch dim
        images = {}
        for k, v in data_in["image"].items():
            arr = jnp.asarray(v)
            if arr.ndim == 3:  # HWC
                arr = arr[None, ...]
            images[k] = arr

        # Ensure image masks have batch dim and bool dtype
        image_mask = {}
        for k, v in data_in["image_mask"].items():
            arr = jnp.asarray(v, dtype=bool)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.shape[0] != batch:
                arr = np.broadcast_to(arr.reshape(1), (batch,))
            image_mask[k] = arr

        state_arr = jnp.asarray(data_in["state"], dtype=jnp.float32)
        if state_arr.ndim == 1:
            state_arr = state_arr[None, ...]
        pad_dim = cfg.model.action_dim - state_arr.shape[-1]
        if pad_dim > 0:
            state_arr = jnp.pad(state_arr, ((0, 0), (0, pad_dim)), mode="constant")
        elif pad_dim < 0:
            state_arr = state_arr[..., : cfg.model.action_dim]
        data_dict = {
            "image": images,
            "image_mask": image_mask,
            "state": state_arr,
            "tokenized_prompt": tok,
            "tokenized_prompt_mask": tok_mask,
        }
        observation = _model.Observation.from_dict(data_dict)

        rng = jax.random.key(int(time.time() * 1e6) & 0xFFFFFFFF)
        # Sample actions for full horizon; take first step.
        actions = np.array(model.sample_actions(rng, observation))[0]
        acted = outputs_tf({"actions": actions})["actions"][0]
        # Safety: clip/finite filter to avoid NaNs or huge torques.
        acted = np.nan_to_num(acted, nan=0.0, posinf=1.0, neginf=-1.0)
        acted = np.clip(acted, -0.5, 0.5)  # tighten if sim is unstable; expand if stable
        return acted

    return policy_fn


def compute_reward(pi_obs: Dict[str, Any], info: Dict[str, Any], action: np.ndarray, env_reward: float = 0.0) -> float:
    """Task-aware reward placeholder (Stage 1).

    If env exposes task info (e.g., cube/goal positions or success flags), use them here.
    For gym_aloha transfer cube, available signals:
      - env_reward in {0..4} (contact-based staging; 4 = successful transfer)
      - info["is_success"] when reward == 4
    We also keep small shaping on state norm and action magnitude.
    """
    dist_term = -float(np.linalg.norm(pi_obs["state"])) * 0.1  # mild shaping; adjust as needed
    act_pen = -float(np.linalg.norm(action)) * 0.01
    env_term = float(env_reward)  # use env's discrete reward directly
    success_bonus = 0.0
    if info is not None:
        for k in ("success", "is_success", "done_success"):
            if k in info and bool(info[k]):
                success_bonus = 5.0
                break
        # If env later provides a distance metric, add it here:
        if "dist_to_goal" in info:
            dist_term = -float(info["dist_to_goal"])
    return env_term + dist_term + act_pen + success_bonus


def rollout(env: AlohaSimEnvironment, policy_fn, episode_steps: int = 200) -> float:
    env.reset()
    total_reward = 0.0
    rewards = []
    obs0 = env.get_observation()
    print(f"[{datetime.now().isoformat()}] reset obs keys: {list(obs0.keys())}")
    if "pixels" in obs0:
        print(f"[{datetime.now().isoformat()}] reset obs pixels keys: {list(obs0['pixels'].keys())}")
    if "images" in obs0:
        print(f"[{datetime.now().isoformat()}] reset obs images keys: {list(obs0['images'].keys())}")

    pi_obs0 = convert_env_obs(obs0)
    action0 = policy_fn(pi_obs0)
    env_r0, info0 = env.apply_action({"actions": action0})
    r0 = compute_reward(pi_obs0, info0 or {}, action0, env_reward=env_r0)
    rewards.append(r0)
    total_reward += r0
    print(
        f"[{datetime.now().isoformat()}] step=0 "
        f"state_norm={np.linalg.norm(pi_obs0['state']):.3f} "
        f"action_min={action0.min():.3f} action_max={action0.max():.3f} "
        f"action_mean={action0.mean():.3f} reward={r0:.3f} "
        f"env_r={env_r0:.3f} info_keys={list(info0.keys()) if info0 else []}"
    )

    for t in tqdm(range(1, episode_steps), desc="Rollout", ncols=100):
        obs = env.get_observation()
        pi_obs = convert_env_obs(obs)
        action = policy_fn(pi_obs)
        if t % 10 == 0:
            print(
                f"[{datetime.now().isoformat()}] step={t} "
                f"state_norm={np.linalg.norm(pi_obs['state']):.3f} "
                f"action_min={action.min():.3f} action_max={action.max():.3f} "
                f"action_mean={action.mean():.3f}"
            )
        env_r, info = env.apply_action({"actions": action})
        if info and t % 20 == 0:
            print(f"[{datetime.now().isoformat()}] info keys: {list(info.keys())}")
        r = compute_reward(pi_obs, info or {}, action, env_reward=env_r)
        rewards.append(r)
        total_reward += r
        if env.is_episode_complete():
            break
    if rewards:
        print(f"[rollout] ep_mean_reward={np.mean(rewards):.3f}, ep_min_reward={np.min(rewards):.3f}")
        print(f"[rollout] ep_reward_hist_first5={np.array(rewards[:5])}")
        if len(rewards) > 10:
            print(f"[rollout] ep_reward_hist_last5={np.array(rewards[-5:])}")
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pi0_aloha_sim_300m_lora_h12_fsdp/aloha_sim_300m_lora_h12_fsdp/199/params",
        help="Path to params checkpoint (params dir).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_aloha_sim_300m_lora_h12_fsdp",
        help="TrainConfig name to build model.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    # Match gym_aloha registration used in examples/aloha_sim/main.py
    parser.add_argument("--task", type=str, default="gym_aloha/AlohaTransferCube-v0")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    policy_fn = make_policy(args.checkpoint, args.config_name)
    env = AlohaSimEnvironment(task=args.task, max_episode_steps=args.max_episode_steps)

    rewards = []
    for ep in range(args.episodes):
        start = time.time()
        ep_ret = rollout(env, policy_fn, episode_steps=args.max_episode_steps)
        rewards.append(ep_ret)
        print(f"Episode {ep}: reward={ep_ret:.3f}, elapsed={time.time() - start:.1f}s")

    print(f"Avg reward over {len(rewards)} episodes: {np.mean(rewards):.3f}")


if __name__ == "__main__":
    main()
"""
Aloha-sim rollout (research stub for π-RL style fine-tuning).

What this script does (current state):
- Reset AlohaSimEnvironment, convert obs to Aloha policy format (images + state + prompt tokens).
- Load a pre-trained VLA (BC) checkpoint (e.g., tiny_b4 or 300m LoRA FSDP) and run sample_actions per step.
- Apply actions to env with basic safety (nan_to_num + clip).
- Compute a toy reward (state norm + action penalty) and log episode stats (every 10 steps + summaries).

What it does NOT do yet (missing pieces for a full π-RL loop):
- No trajectory buffer or advantage/value estimation.
- No policy/value update (actor-critic/PPO/REINFORCE) or gradient steps.
- No task-aware reward (env currently exposes only state/images; no explicit cube/goal/success info here).
- No logging to wandb/TB; no checkpointing of RL-updated params.

Use this as a starting point to:
1) Plug in task-specific reward/success signals (needs env info: cube/goal positions or success flags).
2) Add a trajectory buffer + update_policy skeleton for on-policy RL (tiny model first, then LoRA).
3) Iterate on action scaling/clipping if sim instability appears.

Env signals (from gym_aloha/AlohaTransferCube-v0):
- Observation from env wrapper includes only images (top cam) and agent_pos (qpos of 14 dof).
- Env internal reward: discrete 0..4 based on contacts (0 idle, 1 right touch, 2 right lifted, 3 left touch, 4 successful transfer).
- Env info: {"is_success": reward == 4}. Termination in gym_aloha is tied to reward==4 (terminated); wrapper now treats success/terminated/truncated as done.
=> We lack explicit cube/goal poses; for task-aware reward we can use env_reward and success flag. Distances would need env_state wiring if added later.
"""
