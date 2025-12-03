"""
Policy evaluation helper: run multiple episodes and report success rate, avg length, avg return.

Uses the same policy construction helpers as rl_loop.py. This is still BC-only inference
(no RL updates). Rewards are computed via compute_reward in rl_loop (task-aware placeholder).
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from tqdm import tqdm

from examples.aloha_sim.env import AlohaSimEnvironment
from examples.aloha_sim.rl_loop import make_policy, convert_env_obs, compute_reward

# Default to headless rendering
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat()}] {msg}", flush=True)


def run_episode(env: AlohaSimEnvironment, policy_fn, max_steps: int, ep_idx: int) -> Tuple[float, int, bool, list[float], list[float]]:
    log(f"Episode {ep_idx} start: reset env")
    env.reset()
    total_reward = 0.0
    success = False
    env_rewards = []
    shaped_rewards = []

    for t in range(max_steps):
        t_start = time.time()
        obs = env.get_observation()
        pi_obs = convert_env_obs(obs)
        action = policy_fn(pi_obs)
        env_r, info = env.apply_action({"actions": action})
        r = compute_reward(pi_obs, info or {}, action, env_reward=env_r)
        total_reward += r
        env_rewards.append(env_r)
        shaped_rewards.append(r)
        if t == 0 or t % 20 == 0:
            log(
                f"[ep {ep_idx}] step={t} state_norm={np.linalg.norm(pi_obs['state']):.3f} "
                f"action_min={action.min():.3f} action_max={action.max():.3f} "
                f"action_mean={action.mean():.3f} env_r={env_r:.3f} info_keys={list(info.keys()) if info else []}"
            )
        dt = time.time() - t_start
        if dt > 5.0:
            log(f"[ep {ep_idx}] step={t} took {dt:.2f}s")
        # success flag if available
        if info:
            for k in ("success", "is_success", "done_success"):
                if k in info and bool(info[k]):
                    success = True
                    break
        if env.is_episode_complete():
            break
    return total_reward, t + 1, success, env_rewards, shaped_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params",
        help="Path to params checkpoint (params dir).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_aloha_sim_tiny_b4",
        help="TrainConfig name to build model.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--task", type=str, default="gym_aloha/AlohaTransferCube-v0")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--save-metrics", type=str, default="", help="Optional path to save metrics json.")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    log(f"Loading policy from {args.checkpoint} with config {args.config_name}")
    policy_fn = make_policy(args.checkpoint, args.config_name)
    log("Policy loaded.")
    log(f"Creating env: {args.task}, max_steps={args.max_episode_steps}")
    env = AlohaSimEnvironment(task=args.task, max_episode_steps=args.max_episode_steps)
    log("Env created.")
    # Warmup to trigger JIT and avoid first-step latency in eval loop
    log("Warmup: single forward/step after reset")
    env.reset()
    obs_w = env.get_observation()
    pi_obs_w = convert_env_obs(obs_w)
    action_w = policy_fn(pi_obs_w)
    env_r_w, info_w = env.apply_action({"actions": action_w})
    _ = compute_reward(pi_obs_w, info_w or {}, action_w, env_reward=env_r_w)
    log("Warmup done.")

    returns = []
    lengths = []
    successes = []
    all_env_rewards = []
    all_shaped_rewards = []
    for ep in tqdm(range(args.episodes), desc="Eval Episodes", ncols=100, dynamic_ncols=True):
        ret, steps, succ, env_rs, shaped_rs = run_episode(env, policy_fn, args.max_episode_steps, ep_idx=ep)
        returns.append(ret)
        lengths.append(steps)
        successes.append(succ)
        all_env_rewards.append(env_rs)
        all_shaped_rewards.append(shaped_rs)
        log(f"[{Path(args.checkpoint)}][{ep}] return={ret:.3f}, steps={steps}, success={succ}")

    print("---- Summary ----")
    print(f"Episodes: {len(returns)}")
    print(f"Success rate: {np.mean(successes):.3f}")
    print(f"Avg episode length: {np.mean(lengths):.2f}")
    print(f"Avg return: {np.mean(returns):.3f}")

    if args.save_metrics:
        save_path = Path(args.save_metrics)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "returns": returns,
            "lengths": lengths,
            "successes": successes,
            "env_rewards": all_env_rewards,
            "shaped_rewards": all_shaped_rewards,
            "config": {
                "checkpoint": str(args.checkpoint),
                "config_name": args.config_name,
                "episodes": args.episodes,
                "max_steps": args.max_episode_steps,
            },
        }
        with save_path.open("w") as f:
            json.dump(payload, f, indent=2)
        log(f"Saved metrics to {save_path}")


if __name__ == "__main__":
    main()
