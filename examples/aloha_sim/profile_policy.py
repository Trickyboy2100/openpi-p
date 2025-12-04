"""
Micro-profiler for policy inference.

Runs make_profiled_policy on one frozen observation and reports average
per-stage latency (inputs_tf, tokenize, obs_build, sample_actions, total).
Use for spotting bottlenecks in policy forward.
"""

import argparse
import os
import numpy as np
from statistics import mean

from examples.aloha_sim.env import AlohaSimEnvironment
from examples.aloha_sim.rl_loop import convert_env_obs, make_profiled_policy

# Headless rendering for mujoco
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def main():
    parser = argparse.ArgumentParser(description="Profile policy forward pass.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params",
        help="Path to params checkpoint (params dir).",
    )
    parser.add_argument("--config-name", type=str, default="pi0_aloha_sim_tiny_b4")
    parser.add_argument("--task", type=str, default="gym_aloha/AlohaTransferCube-v0")
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--iters", type=int, default=6, help="Total calls including warmup.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of initial calls to discard.")
    args = parser.parse_args()

    profiled_policy = make_profiled_policy(args.checkpoint, args.config_name)
    import jax
    print(f"[jax-debug] devices: {jax.devices()}")
    env = AlohaSimEnvironment(task=args.task, max_episode_steps=args.max_episode_steps)
    env.reset()
    obs = env.get_observation()
    obs_np = convert_env_obs(obs)
    # Debug: print image shapes
    for k, v in obs_np["images"].items():
        print(f"[debug] obs_np images[{k}] shape={np.asarray(v).shape}")

    timings = []
    for i in range(args.iters):
        action, tdict = profiled_policy(obs_np)
        _ = np.asarray(action)  # touch array to avoid lazy eval concerns
        if i >= args.warmup:
            timings.append(tdict)

    if not timings:
        print("No timings collected (iters <= warmup).")
        return

    keys = ["inputs_dt", "tokenize_dt", "obs_build_dt", "sample_actions_dt", "total_dt"]
    avg = {k: mean([t.get(k, 0.0) for t in timings]) for k in keys}
    print("---- Policy forward profiling (averages) ----")
    for k in keys:
        print(f"{k}: {avg[k]:.6f}s")


if __name__ == "__main__":
    main()
