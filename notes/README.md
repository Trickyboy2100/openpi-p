# Personal Notes for ALOHA-sim π-RL (timestamp: 2025-12-03T20:58:07+08:00)

## Repo / Env
- Path: `~/Gitclones/openpi`
- venv: `source .venv/bin/activate`
- GPU env vars often used: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`

## Checkpoints of interest
- Tiny (multi-GPU): `checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params`
- FSDP LoRA 300m (h12, token24): `checkpoints/pi0_aloha_sim_300m_lora_h12_fsdp/aloha_sim_300m_lora_h12_fsdp/199/params`

## Training configs (key ones)
- `pi0_aloha_sim_300m_lora_h12_fsdp` (LoRA, gemma_300m_lora, horizon=12, token_len=24, batch=4, fsdp_devices=4)
  - Train cmd example:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
      python scripts/train.py pi0_aloha_sim_300m_lora_h12_fsdp \
        --exp-name=aloha_sim_300m_lora_h12_fsdp \
        --overwrite
    ```
- Tiny configs: `pi0_aloha_sim_tiny`, `pi0_aloha_sim_tiny_b4` (dummy models, small action_horizon)

## RL loop (current state)
- File: `examples/aloha_sim/rl_loop.py`
- Does: env reset → obs convert → policy forward (BC checkpoint) → action safety (nan_to_num + clip) → apply_action → toy reward logging.
- Reward placeholder: `-(||state|| + 0.01*||action||)`, optional success_bonus if info has success flag.
- Logging: timestep logs every 10 steps with state/action stats; episode reward stats printed.
- Not implemented yet: trajectory buffer, advantage/value head, policy updates (PPO/REINFORCE), true task reward (needs env goal/success info).
- Run example (FSDP LoRA):
  ```bash
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  CUDA_VISIBLE_DEVICES=0 \
    python examples/aloha_sim/rl_loop.py \
      --checkpoint checkpoints/pi0_aloha_sim_300m_lora_h12_fsdp/aloha_sim_300m_lora_h12_fsdp/199/params \
      --config-name pi0_aloha_sim_300m_lora_h12_fsdp \
      --episodes 1
  ```
- For faster iteration: swap checkpoint/config to tiny_b4 and reduce `--max-episode-steps`.

## Next steps (personal TODO)
- Plug in task-specific reward: need cube/goal pose or success flag from env info; replace compute_reward accordingly.
  - Add success bonus (+5/+10), distance term to goal, keep small action penalty.
- Add minimal RL skeleton: trajectory buffer, optional value/residual head, update_policy (e.g., REINFORCE) on tiny model first.
- Reduce episode length when debugging to cut compile/run time (e.g., 50–100 steps).
- If pushing to origin fails (network), retry from a networked terminal: `git push origin HEAD`.
