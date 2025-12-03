"""
Skeleton for future RL fine-tuning on ALOHA-sim (π-RL style).
NOT runnable yet; provides structure and TODOs.

Plan (for Stage 2):
- Reuse convert_env_obs / make_policy to collect trajectories.
- Keep backbone frozen; optionally add small trainable heads (value/residual or LoRA modules).
- On-policy (REINFORCE/PPO-lite) with small batch of episodes.
"""

# TODO: implement trajectory buffer (s, a, r, done, info)
# class TrajectoryBuffer:
#     def add(self, obs, action, reward, done, info): ...
#     def get(self): return stacked arrays

# TODO: add value head / critic wrapper around existing policy outputs.
# def add_value_head(model): ...

# TODO: advantage/returns computation (e.g., GAE or simple returns)
# def compute_advantages(rewards, dones, values): ...

# TODO: update step (e.g., REINFORCE or PPO-lite)
# def update_policy(params, trajectories, optimizer): ...

# This file is a placeholder to align code structure for Stage 2.
