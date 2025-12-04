"""
Stage 2: Minimal REINFORCE fine-tuning on π0 tiny_b4 for AlohaTransferCube-v0.

Purpose:
- Provide a runnable on-policy REINFORCE baseline on top of the BC policy.
- Reuse Stage 1 bridges: AlohaSimEnvironment, convert_env_obs, make_policy, compute_reward.
- Keep backbone frozen; only update a small residual Gaussian head on actions.

Simplifications:
- Baseline=0 (simple Monte Carlo return), no value/GAE yet.
- Trainable params: a tiny linear residual head (state -> action_mean offset) + log_std.
- Sampled actions ~ N(base_action + residual, std), loss = -return * logprob(action).
- compute_reward uses shaped reward (env_reward + success bonus + small penalties).

Future extensions (TODO):
- Add value baseline/GAE, richer heads/LoRA, better action scaling/clipping, entropy bonus.

Example:
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    python examples/aloha_sim/rl_train_pi0_tiny.py \\
      --checkpoint checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params \\
      --config-name pi0_aloha_sim_tiny_b4 \\
      --episodes-per-iter 3 \\
      --train-iters 5 \\
      --max-episode-steps 50

Debug-fast config (for correctness/profiling, not for success rate):
- episodes-per-iter=1
- train-iters=2
- max-episode-steps=10
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
import time
import json

import jax
import jax.numpy as jnp
import optax
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
    log_probs: List[float]
    observations: List[Dict[str, Any]]
    infos: List[Dict[str, Any]]
    success: bool
    policy_dt: float
    env_dt: float


def init_head_params(state_dim: int, action_dim: int, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
    """Tiny residual head: mean = base_action + state @ W + b; log_std is a scalar."""
    k1, k2 = jax.random.split(key)
    w = jax.random.normal(k1, (state_dim, action_dim)) * 0.01
    b = jnp.zeros((action_dim,))
    log_std = jnp.array(-1.0)  # std ~ 0.37
    return {"w": w, "b": b, "log_std": log_std}


def policy_with_head(
    base_action: jnp.ndarray,
    state: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, float]:
    """Compute final action and logprob under Gaussian residual head."""
    residual_mean = state @ params["w"] + params["b"]
    mean = base_action + residual_mean
    std = jnp.exp(params["log_std"])
    eps = jax.random.normal(key, mean.shape)
    action = mean + std * eps
    # Gaussian logprob
    log_prob = -0.5 * jnp.sum(((action - mean) / std) ** 2 + 2 * params["log_std"] + jnp.log(2 * jnp.pi))
    # Safety clip (keep consistent with rollout)
    action = jnp.clip(action, -0.5, 0.5)
    return action, log_prob


def collect_trajectory(
    env: AlohaSimEnvironment,
    base_policy_fn,
    head_params: Dict[str, jnp.ndarray],
    max_steps: int,
    rng: jax.random.PRNGKey,
    allow_contact_shaping: bool,
) -> Tuple[Trajectory, jax.random.PRNGKey]:
    """Collect one episode using current policy (base + residual head)."""
    env.reset()
    rewards = []
    env_rewards = []
    actions = []
    log_probs = []
    observations = []
    infos = []
    success = False
    policy_time = 0.0
    env_time = 0.0
    for t in range(max_steps):
        rng, k1 = jax.random.split(rng)
        obs = env.get_observation()
        pi_obs = convert_env_obs(obs)
        # base action
        t0 = time.perf_counter()
        base_action = jnp.asarray(base_policy_fn(pi_obs), dtype=jnp.float32)
        policy_time += time.perf_counter() - t0
        state_vec = jnp.asarray(pi_obs["state"], dtype=jnp.float32)
        if state_vec.ndim == 1:
            state_vec = state_vec[None, ...]
        state_vec = state_vec[0]  # (S,)
        action, logp = policy_with_head(base_action, state_vec, head_params, k1)
        t1 = time.perf_counter()
        env_r, info = env.apply_action({"actions": np.array(action)})
        env_time += time.perf_counter() - t1
        r = compute_reward(pi_obs, info or {}, np.array(action), env_reward=env_r, allow_contact_shaping=allow_contact_shaping)
        rewards.append(r)
        env_rewards.append(env_r)
        actions.append(np.array(action))
        log_probs.append(float(logp))
        observations.append(pi_obs)
        infos.append(info or {})
        if info and any(info.get(k, False) for k in ("success", "is_success", "done_success")):
            success = True
        if env.is_episode_complete():
            break
    # Monte Carlo return (baseline=0)
    ret = float(np.sum(rewards))
    traj = Trajectory(
        returns=[ret],
        rewards=rewards,
        env_rewards=env_rewards,
        actions=actions,
        log_probs=log_probs,
        observations=observations,
        infos=infos,
        success=success,
        policy_dt=policy_time,
        env_dt=env_time,
    )
    return traj, rng


def collect_batch(
    env: AlohaSimEnvironment,
    base_policy_fn,
    head_params: Dict[str, jnp.ndarray],
    episodes: int,
    max_steps: int,
    rng: jax.random.PRNGKey,
    allow_contact_shaping: bool,
) -> Tuple[List[Trajectory], jax.random.PRNGKey]:
    batch = []
    for ep in range(episodes):
        traj, rng = collect_trajectory(env, base_policy_fn, head_params, max_steps, rng, allow_contact_shaping)
        batch.append(traj)
        print(
            f"[{datetime.now().isoformat()}][collect] ep={ep} return={traj.returns[0]:.3f} "
            f"success={traj.success} steps={len(traj.rewards)} "
            f"env_r_pos_frac={(np.array(traj.env_rewards) > 0).mean():.3f}"
        )
    return batch, rng


def compute_discounted_returns(traj: Trajectory, gamma: float) -> np.ndarray:
    """Compute discounted returns per step."""
    rewards = np.array(traj.rewards, dtype=np.float32)
    g = 0.0
    rets = []
    for r in rewards[::-1]:
        g = float(r) + gamma * g
        rets.append(g)
    return np.array(rets[::-1], dtype=np.float32)


def prepare_batch(trajs: List[Trajectory], gamma: float) -> Dict[str, np.ndarray]:
    """Flatten trajectories into arrays for REINFORCE loss."""
    returns = []
    logps = []
    for traj in trajs:
        traj_rets = compute_discounted_returns(traj, gamma)
        returns.append(traj_rets)
        logps.append(np.array(traj.log_probs, dtype=np.float32))
    return {
        "returns": np.concatenate(returns, axis=0) if returns else np.array([], dtype=np.float32),
        "logps": np.concatenate(logps, axis=0) if logps else np.array([], dtype=np.float32),
    }


def update_policy(
    head_params: Dict[str, jnp.ndarray],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch: Dict[str, np.ndarray],
) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, float]:
    """REINFORCE update on residual head: loss = -E[adv * logp], adv = normalized returns."""
    returns = jnp.asarray(batch["returns"])
    logps = jnp.asarray(batch["logps"])
    if returns.size == 0:
        return head_params, opt_state, 0.0
    mean = jnp.mean(returns)
    std = jnp.std(returns) + 1e-6
    advantages = (returns - mean) / std

    def loss_fn(params):
        return -jnp.mean(advantages * logps)

    loss, grads = jax.value_and_grad(loss_fn)(head_params)
    updates, opt_state = optimizer.update(grads, opt_state, head_params)
    new_params = optax.apply_updates(head_params, updates)
    return new_params, opt_state, float(loss)


def evaluate(
    env: AlohaSimEnvironment,
    base_policy_fn,
    head_params: Dict[str, jnp.ndarray],
    episodes: int,
    max_steps: int,
    rng: jax.random.PRNGKey,
    allow_contact_shaping: bool,
) -> Tuple[Dict[str, float], jax.random.PRNGKey]:
    """Lightweight eval: success rate, avg env_reward, avg shaped return."""
    successes = []
    env_returns = []
    shaped_returns = []
    for ep in range(episodes):
        traj, rng = collect_trajectory(env, base_policy_fn, head_params, max_steps, rng, allow_contact_shaping)
        successes.append(traj.success)
        env_returns.append(float(np.sum(traj.env_rewards)))
        shaped_returns.append(traj.returns[0])
    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_env_return": float(np.mean(env_returns)) if env_returns else 0.0,
        "avg_shaped_return": float(np.mean(shaped_returns)) if shaped_returns else 0.0,
    }, rng


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 minimal RL fine-tuning skeleton (π0 tiny_b4).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to BC checkpoint params.")
    parser.add_argument("--config-name", type=str, default="pi0_aloha_sim_tiny_b4", help="TrainConfig name.")
    parser.add_argument("--episodes-per-iter", type=int, default=5)
    parser.add_argument("--train-iters", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--task", type=str, default="gym_aloha/AlohaTransferCube-v0")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount for returns.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for residual head.")
    parser.add_argument("--log-json", type=str, default="", help="Optional path to append per-iter JSON logs.")
    parser.add_argument("--allow-contact-shaping", action="store_true", default=True, help="Enable contact-based reward shaping.")
    parser.add_argument("--stage3-debug", action="store_true", help="Use Stage3 tiny debug defaults (episodes-per-iter=2, train-iters=5, max-episode-steps=20, gamma=0.97).")
    args = parser.parse_args()

    if args.stage3_debug:
        args.episodes_per_iter = 2
        args.train_iters = 5
        args.max_episode_steps = 20
        args.gamma = 0.97
        print("[debug] Stage3 tiny debug config enabled: episodes-per-iter=2, train-iters=5, max-episode-steps=20, gamma=0.97")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"[init] loading policy from {checkpoint} config={args.config_name}")
    base_policy_fn = make_policy(str(checkpoint), args.config_name)
    env = AlohaSimEnvironment(task=args.task, max_episode_steps=args.max_episode_steps)
    print("[init] env created.")
    import jax
    print(f"[jax-debug] devices: {jax.devices()}")

    # Initialize residual head and optimizer
    env.reset()
    obs0 = env.get_observation()
    sample_action = jnp.asarray(base_policy_fn(convert_env_obs(obs0)), dtype=jnp.float32)
    try:
        print(f"[jax-debug] base_policy sample_action device: {sample_action.device()}")
    except Exception:
        pass
    action_dim = sample_action.shape[-1]
    # state is padded to action_dim (=32) in preprocessing
    state_dim = action_dim
    rng = jax.random.PRNGKey(0)
    head_params = init_head_params(state_dim, action_dim, rng)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(head_params)

    log_path = Path(args.log_json) if args.log_json else None

    for it in range(args.train_iters):
        print(f"[{datetime.now().isoformat()}] === Train iter {it} ===")
        trajs, rng = collect_batch(
            env,
            base_policy_fn,
            head_params,
            args.episodes_per_iter,
            args.max_episode_steps,
            rng,
            args.allow_contact_shaping,
        )
        batch = prepare_batch(trajs, gamma=args.gamma)
        head_params, opt_state, loss = update_policy(head_params, optimizer, opt_state, batch)
        eval_stats, rng = evaluate(
            env,
            base_policy_fn,
            head_params,
            args.eval_episodes,
            args.max_episode_steps,
            rng,
            args.allow_contact_shaping,
        )
        total_steps = np.sum([len(t.rewards) for t in trajs]) or 1
        mean_policy_dt = np.sum([t.policy_dt for t in trajs]) / total_steps
        mean_env_dt = np.sum([t.env_dt for t in trajs]) / total_steps
        success_rate_batch = float(np.mean([t.success for t in trajs]))
        avg_env_r = float(np.mean([np.sum(t.env_rewards) for t in trajs]))
        avg_shaped_ret = float(np.mean([t.returns[0] for t in trajs]))
        print(
            f"[{datetime.now().isoformat()}][train] iter={it} loss={loss:.4f} "
            f"avg_len={np.mean([len(t.rewards) for t in trajs]):.2f} "
            f"avg_env_r={avg_env_r:.3f} "
            f"avg_shaped_ret={avg_shaped_ret:.3f} "
            f"success_rate_batch={success_rate_batch:.3f}"
        )
        print(f"[{datetime.now().isoformat()}][eval] iter={it} stats={eval_stats}")
        print(
            f"[profile] iter={it} mean_policy_step_dt={mean_policy_dt:.6f}s "
            f"mean_env_step_dt={mean_env_dt:.6f}s"
        )
        if log_path:
            payload = {
                "iter": it,
                "batch_success_rate": success_rate_batch,
                "batch_avg_env_r": avg_env_r,
                "batch_avg_shaped_ret": avg_shaped_ret,
                "eval_stats": eval_stats,
                "mean_policy_step_dt": mean_policy_dt,
                "mean_env_step_dt": mean_env_dt,
            }
            if log_path.exists():
                with log_path.open("r") as f:
                    data = json.load(f)
            else:
                data = {"records": []}
            if "records" not in data or not isinstance(data["records"], list):
                data["records"] = []
            data["records"].append(payload)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as f:
                json.dump(data, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

# Multi-experiment (multi-GPU) launcher notes:
# - Run separate processes pinned to different GPUs via CUDA_VISIBLE_DEVICES.
# - Example grid (vary lr/gamma); keep MUJOCO_GL/PYOPENGL_PLATFORM for headless sim.
#   # GPU0: small LR
#   CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
#     python examples/aloha_sim/rl_train_pi0_tiny.py --checkpoint .../199/params --config-name pi0_aloha_sim_tiny_b4 \
#     --episodes-per-iter 3 --train-iters 5 --max-episode-steps 50 --lr 1e-4
#   # GPU1: medium LR
#   CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
#     python examples/aloha_sim/rl_train_pi0_tiny.py --checkpoint .../199/params --config-name pi0_aloha_sim_tiny_b4 \
#     --episodes-per-iter 3 --train-iters 5 --max-episode-steps 50 --lr 3e-4
#   # GPU2: higher LR
#   CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
#     python examples/aloha_sim/rl_train_pi0_tiny.py --checkpoint .../199/params --config-name pi0_aloha_sim_tiny_b4 \
#     --episodes-per-iter 3 --train-iters 5 --max-episode-steps 50 --lr 1e-3
#   # GPU3: different gamma
#   CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
#     python examples/aloha_sim/rl_train_pi0_tiny.py --checkpoint .../199/params --config-name pi0_aloha_sim_tiny_b4 \
#     --episodes-per-iter 3 --train-iters 5 --max-episode-steps 50 --gamma 0.97

# Profiling interpretation (from [profile] line):
# - If mean_policy_step_dt dominates: consider jax.jit/warmup on fixed shapes for policy_fn,
#   cache tokenizer/prompt tokens, or batch multiple obs for forward.
# - If mean_env_step_dt dominates: reduce max_episode_steps for quick tests, lower render cost,
#   or parallelize env processes (multi-proc) to amortize step latency.
