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

    def ensure_chw(img: Any, cam_name: str) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        # If HWC (channel last), move to CHW; if already CHW, keep; if 2D, add singleton channel.
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 2:
            arr = np.transpose(arr[..., None], (2, 0, 1))
        if arr.ndim != 3 or arr.shape[0] not in (1, 3):
            raise ValueError(f"[convert_env_obs] Unexpected image shape for {cam_name}: {arr.shape}")
        return arr

    obs_out = {
        "state": np.asarray(state, dtype=np.float32),
        "images": {
            "cam_high": ensure_chw(top, "cam_high"),  # [C,H,W] uint8
            "cam_left_wrist": ensure_chw(pixels.get("left_wrist", top), "cam_left_wrist"),
            "cam_right_wrist": ensure_chw(pixels.get("right_wrist", top), "cam_right_wrist"),
        },
        "prompt": "Transfer cube",  # default prompt used in training
    }
    if "cube_pos" in gym_obs:
        obs_out["cube_pos"] = np.asarray(gym_obs["cube_pos"], dtype=np.float32)
    return obs_out


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
    cached_prompt: str | None = None
    cached_tokens: jnp.ndarray | None = None
    cached_mask: jnp.ndarray | None = None

    def tokenize_prompt(prompt: str, batch: int) -> tuple[np.ndarray, np.ndarray]:
        nonlocal cached_prompt, cached_tokens, cached_mask
        if cached_prompt == prompt and cached_tokens is not None and cached_mask is not None:
            return cached_tokens, cached_mask
        tokens, mask = tokenizer.tokenize(prompt)
        tokens = jnp.asarray(tokens, dtype=jnp.int32)[None, ...].repeat(batch, axis=0)
        mask = jnp.asarray(mask, dtype=bool)[None, ...].repeat(batch, axis=0)
        cached_prompt, cached_tokens, cached_mask = prompt, tokens, mask
        return tokens, mask

    first_device_log: list[bool] = [False]

    def policy_fn(obs_np: Dict[str, Any]) -> np.ndarray:
        data_in = inputs_tf(obs_np)  # maps to dict with image/state keys expected by model transforms
        batch = 1
        tok, tok_mask = tokenize_prompt(obs_np.get("prompt", ""), batch)

        def ensure_nhwc(x: Any) -> np.ndarray:
            arr = np.asarray(x)
            if arr.ndim == 3:
                if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
            elif arr.ndim == 4:
                if arr.shape[-1] not in (1, 3) and arr.shape[1] in (1, 3):
                    arr = np.transpose(arr, (0, 2, 3, 1))  # NCHW -> NHWC
            if arr.ndim >= 3 and arr.shape[-1] not in (1, 3):
                # Fallback: drop extra channels if shape exploded (e.g., [..., 224])
                arr = arr[..., :3]
            return arr

        # Ensure images have batch dim and NHWC
        images = {}
        for k, v in data_in["image"].items():
            arr = ensure_nhwc(v)
            if arr.ndim == 3:  # HWC
                arr = arr[None, ...]
            images[k] = jnp.asarray(arr)

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
        actions_jax = model.sample_actions(rng, observation)
        if not first_device_log[0]:
            try:
                print(f"[jax-debug] policy devices={jax.devices()}")
                print(f"[jax-debug] sample_actions device={actions_jax.device()}")
            except Exception:
                pass
            first_device_log[0] = True
        actions = np.array(actions_jax)[0]
        acted = outputs_tf({"actions": actions})["actions"][0]
        # Safety: clip/finite filter to avoid NaNs or huge torques.
        acted = np.nan_to_num(acted, nan=0.0, posinf=1.0, neginf=-1.0)
        acted = np.clip(acted, -0.5, 0.5)  # tighten if sim is unstable; expand if stable
        return acted

    return policy_fn


def make_profiled_policy(checkpoint_path: str, config_name: str):
    """Profiled variant of make_policy: returns fn(obs)->(action, timing_dict)."""
    cfg = get_config(config_name or "pi0_aloha_sim_tiny_b4")
    params_pure = _model.restore_params(checkpoint_path, restore_type=np.ndarray)
    model = cfg.model.load(params_pure)
    model.eval()

    inputs_tf = AlohaInputs(adapt_to_pi=True)
    outputs_tf = AlohaOutputs(adapt_to_pi=True)
    tokenizer = PaligemmaTokenizer(cfg.model.max_token_len)
    cached_prompt: str | None = None
    cached_tokens: jnp.ndarray | None = None
    cached_mask: jnp.ndarray | None = None

    def tokenize_prompt(prompt: str, batch: int, profile: bool = False, timings: Dict[str, float] | None = None):
        nonlocal cached_prompt, cached_tokens, cached_mask
        t0 = time.perf_counter() if profile else 0.0
        if cached_prompt == prompt and cached_tokens is not None and cached_mask is not None:
            tokens, mask = cached_tokens, cached_mask
        else:
            tokens_np, mask_np = tokenizer.tokenize(prompt)
            tokens = jnp.asarray(tokens_np, dtype=jnp.int32)[None, ...].repeat(batch, axis=0)
            mask = jnp.asarray(mask_np, dtype=bool)[None, ...].repeat(batch, axis=0)
            cached_prompt, cached_tokens, cached_mask = prompt, tokens, mask
        if profile and timings is not None:
            timings["tokenize_dt"] = time.perf_counter() - t0
        return tokens, mask

    printed_shape = False

    def build_observation(obs_np: Dict[str, Any], profile: bool = False) -> tuple[_model.Observation, Dict[str, float]]:
        timings: Dict[str, float] = {}
        t0 = time.perf_counter() if profile else 0.0
        data_in = inputs_tf(obs_np)
        if profile:
            timings["inputs_dt"] = time.perf_counter() - t0
        batch = 1
        tok, tok_mask = tokenize_prompt(obs_np.get("prompt", ""), batch, profile=profile, timings=timings)

        def ensure_nhwc(x: Any) -> np.ndarray:
            arr = np.asarray(x)
            if arr.ndim == 3:
                if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
            elif arr.ndim == 4:
                if arr.shape[-1] not in (1, 3) and arr.shape[1] in (1, 3):
                    arr = np.transpose(arr, (0, 2, 3, 1))  # NCHW -> NHWC
            return arr

        images = {}
        for k, v in data_in["image"].items():
            arr = ensure_nhwc(v)
            if arr.ndim == 3:
                arr = arr[None, ...]
            images[k] = jnp.asarray(arr)

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
        t_obs = time.perf_counter() if profile else 0.0
        observation = _model.Observation.from_dict(data_dict)
        nonlocal printed_shape
        if not printed_shape:
            printed_shape = True
            for k, v in images.items():
                print(f"[debug] data_in image {k} shape {v.shape}")
        if profile:
            timings["obs_build_dt"] = time.perf_counter() - t_obs
        return observation, timings

    first_device_log: list[bool] = [False]

    def profiled_policy_fn(obs_np: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, float]]:
        total_start = time.perf_counter()
        observation, timings = build_observation(obs_np, profile=True)
        rng = jax.random.key(int(time.time() * 1e6) & 0xFFFFFFFF)
        t_sample = time.perf_counter()
        actions_jax = model.sample_actions(rng, observation)
        if not first_device_log[0]:
            try:
                print(f"[jax-debug] policy devices={jax.devices()}")
                print(f"[jax-debug] sample_actions device={actions_jax.device()}")
            except Exception:
                pass
            first_device_log[0] = True
        actions = np.array(actions_jax)[0]
        timings["sample_actions_dt"] = time.perf_counter() - t_sample
        acted = outputs_tf({"actions": actions})["actions"][0]
        acted = np.nan_to_num(acted, nan=0.0, posinf=1.0, neginf=-1.0)
        acted = np.clip(acted, -0.5, 0.5)
        timings["total_dt"] = time.perf_counter() - total_start
        return acted, timings

    return profiled_policy_fn


def compute_reward(
    pi_obs: Dict[str, Any],
    info: Dict[str, Any],
    action: np.ndarray,
    env_reward: float = 0.0,
    allow_contact_shaping: bool = True,
    cube_dist_lambda: float = 0.0,
) -> float:
    """Task-aware reward placeholder (Stage 1).

    If env exposes task info (e.g., cube/goal positions or success flags), use them here.
    For gym_aloha transfer cube, available signals:
      - env_reward in {0..4} (contact-based staging; 4 = successful transfer)
      - info["is_success"] when reward == 4
    We also keep small shaping on state norm and action magnitude.
    """
    dist_term = -float(np.linalg.norm(pi_obs["state"])) * 0.1  # mild shaping; adjust as needed
    act_pen = -float(np.linalg.norm(action)) * 0.01
    if allow_contact_shaping:
        contact_map = [0.0, 0.2, 0.5, 1.0, 3.0]
        env_idx = int(np.clip(env_reward, 0, len(contact_map) - 1))
        env_term = float(contact_map[env_idx])
    else:
        env_term = float(env_reward)  # use env's discrete reward directly
    # Optional distance shaping on cube pose (pos+quat first 7 dims). TODO: wire actual goal from env/info.
    cube_shaping = 0.0
    cube_dist = None
    if cube_dist_lambda > 0 and "cube_pos" in pi_obs:
        cube_pos = np.asarray(pi_obs["cube_pos"], dtype=np.float32)
        cube_goal = None
        # TODO: populate cube_goal from env/info if available.
        if info and "cube_goal" in info:
            cube_goal = np.asarray(info["cube_goal"], dtype=np.float32)
        elif info and "cube_goal_pos" in info:
            cube_goal = np.asarray(info["cube_goal_pos"], dtype=np.float32)
        # Fallback: zero goal placeholder.
        if cube_goal is None:
            cube_goal = np.zeros_like(cube_pos)
        cube_dist = float(np.linalg.norm(cube_pos - cube_goal))
        cube_shaping = -cube_dist_lambda * cube_dist
    success_bonus = 0.0
    if info is not None:
        for k in ("success", "is_success", "done_success"):
            if k in info and bool(info[k]):
                success_bonus = 5.0
                break
        # If env later provides a distance metric, add it here:
        if "dist_to_goal" in info:
            dist_term = -float(info["dist_to_goal"])
    reward_components = {
        "env_term": env_term,
        "dist_term": dist_term,
        "act_pen": act_pen,
        "success_bonus": success_bonus,
        "cube_shaping": cube_shaping,
    }
    if cube_dist is not None:
        reward_components["cube_dist"] = cube_dist
    # Optionally attach to info for downstream logging.
    if info is not None:
        info.setdefault("reward_components", reward_components)
    return env_term + dist_term + act_pen + success_bonus + cube_shaping


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
