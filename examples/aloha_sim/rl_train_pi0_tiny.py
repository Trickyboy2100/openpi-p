"""
Stage 2: Minimal RL fine-tuning skeleton for π0 tiny_b4 on AlohaTransferCube-v0.

Purpose:
- Provide a runnable scaffold for on-policy RL (e.g., REINFORCE-style) on top of the BC policy.
- Reuse Stage 1 bridges: AlohaSimEnvironment, convert_env_obs, make_policy, compute_reward.
- Keep backbone frozen; future work can update only small heads/LoRA blocks.

Simplifications (current file):
- No actual gradient updates are implemented yet (update_policy is a no-op placeholder).
- Returns are baseline=0 (simple Monte Carlo). Advantage/value heads are TODO.
- Uses compute_reward from rl_loop (task-aware placeholder using env_reward + success).

Suggested next steps (not implemented here):
- Add a small trainable head (e.g., residual MLP on actions) or LoRA modules.
- Implement log-prob and gradient updates (REINFORCE / PPO-lite) on the trainable subset.
- Add trajectory buffer + optimizer.

Example run (collection + dummy updates + eval):
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    python examples/aloha_sim/rl_train_pi0_tiny.py \\
      --checkpoint checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params \\
      --config-name pi0_aloha_sim_tiny_b4 \\
      --episodes-per-iter 5 \\
      --train-iters 2 \\
      --max-episode-steps 50
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import numpy as np

from examples.aloha_sim.env import AlohaSimEnvironment
from examples.aloha_sim.rl_loop import convert_env_obs, make_policy, compute_reward

# Headless rendering for mujoco
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


@dataclass
class Trajectory:
    returns: List[float]
    rewards: List[float]
    env_rewards: List[float]
    actions: List[np.ndarray]
    observations: List[Dict[str, Any]]
    infos: List[Dict[str, Any]]
    success: bool


def collect_trajectory(env: AlohaSimEnvironment, policy_fn, max_steps: int) -> Trajectory:
    """Collect one episode using current policy_fn; no gradients here."""
    env.reset()
    rewards = []
    env_rewards = []
    actions = []
    observations = []
    infos = []
    success = False

    for _ in range(max_steps):
        obs = env.get_observation()
        pi_obs = convert_env_obs(obs)
        action = policy_fn(pi_obs)
        env_r, info = env.apply_action({"actions": action})
        r = compute_reward(pi_obs, info or {}, action, env_reward=env_r)
        rewards.append(r)
        env_rewards.append(env_r)
        actions.append(action)
        observations.append(pi_obs)
        infos.append(info or {})
        if info and any(info.get(k, False) for k in ("success", "is_success", "done_success")):
            success = True
        if env.is_episode_complete():
            break
    # Monte Carlo return (baseline=0)
    ret = float(np.sum(rewards))
    return Trajectory(
        returns=[ret],
        rewards=rewards,
        env_rewards=env_rewards,
        actions=actions,
        observations=observations,
        infos=infos,
        success=success,
    )


def collect_batch(env: AlohaSimEnvironment, policy_fn, episodes: int, max_steps: int) -> List[Trajectory]:
    batch = []
    for ep in range(episodes):
        traj = collect_trajectory(env, policy_fn, max_steps)
        batch.append(traj)
        print(
            f"[collect] ep={ep} return={traj.returns[0]:.3f} "
            f"success={traj.success} steps={len(traj.rewards)} "
            f"env_r_pos_frac={(np.array(traj.env_rewards) > 0).mean():.3f}"
        )
    return batch


def dummy_update_policy(policy_fn, trajectories: List[Trajectory]):
    """Placeholder for future RL updates. Currently no-ops."""
    # TODO: Implement log-prob computation and gradient step on trainable head/LoRA.
    print("[update] dummy update (no-op).")
    return policy_fn


def evaluate(env: AlohaSimEnvironment, policy_fn, episodes: int, max_steps: int) -> Dict[str, float]:
    """Lightweight eval: success rate, avg env_reward, avg shaped return."""
    successes = []
    env_returns = []
    shaped_returns = []
    for ep in range(episodes):
        traj = collect_trajectory(env, policy_fn, max_steps)
        successes.append(traj.success)
        env_returns.append(float(np.sum(traj.env_rewards)))
        shaped_returns.append(traj.returns[0])
    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_env_return": float(np.mean(env_returns)) if env_returns else 0.0,
        "avg_shaped_return": float(np.mean(shaped_returns)) if shaped_returns else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 minimal RL fine-tuning skeleton (π0 tiny_b4).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to BC checkpoint params.")
    parser.add_argument("--config-name", type=str, default="pi0_aloha_sim_tiny_b4", help="TrainConfig name.")
    parser.add_argument("--episodes-per-iter", type=int, default=5)
    parser.add_argument("--train-iters", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--task", type=str, default="gym_aloha/AlohaTransferCube-v0")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"[init] loading policy from {checkpoint} config={args.config_name}")
    policy_fn = make_policy(str(checkpoint), args.config_name)
    env = AlohaSimEnvironment(task=args.task, max_episode_steps=args.max_episode_steps)
    print("[init] env created.")

    for it in range(args.train_iters):
        print(f"=== Train iter {it} ===")
        trajs = collect_batch(env, policy_fn, args.episodes_per_iter, args.max_episode_steps)
        policy_fn = dummy_update_policy(policy_fn, trajs)
        eval_stats = evaluate(env, policy_fn, args.eval_episodes, args.max_episode_steps)
        print(f"[eval] iter={it} stats={eval_stats}")

    print("Done.")


if __name__ == "__main__":
    main()
