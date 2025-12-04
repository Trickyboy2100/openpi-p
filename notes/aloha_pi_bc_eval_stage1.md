# Stage 1: π0 Tiny_B4 BC Evaluation (AlohaTransferCube-v0)

Timestamp: 2025-12-04T01:01:03+08:00

## Env & Reward Signals
- Env: `gym_aloha/AlohaTransferCube-v0` (via `examples/aloha_sim/env.py` wrapper).
- Observations (pixels_agent_pos): top camera + `agent_pos` (14-dof qpos). No explicit cube/goal pose exposed.
- Env reward (contact-based, 0..4):
  - 1: right gripper touches cube
  - 2: right gripper lifts cube (not on table)
  - 3: left gripper touches cube
  - 4: left gripper holds cube off table (successful transfer) ⇒ `info["is_success"]=True`, terminated.
- Wrapper `apply_action` treats terminated/truncated/success as done.

## Policy & Scripts
- Policy: π0 tiny_b4 (BC only, backbone frozen), checkpoint `checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params`.
- Eval script: `examples/aloha_sim/eval_policy.py` → metrics saved via `--save-metrics`.
- Reward shaping in `compute_reward` (Stage 1 placeholder):
  - `env_reward` (0..4) +
  - light state norm penalty (−0.1‖state‖) +
  - action penalty (−0.01‖a‖) +
  - success bonus (+5 if `is_success`).

## Eval Config & Command
- Episodes: 20, max_episode_steps: 50.
- Command:
  ```bash
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python examples/aloha_sim/eval_policy.py \
      --checkpoint checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params \
      --config-name pi0_aloha_sim_tiny_b4 \
      --episodes 20 \
      --max-episode-steps 50 \
      --save-metrics outputs/tiny_b4_eval_20ep_50steps.json
  ```

## Quantitative Results (from `outputs/tiny_b4_eval_20ep_50steps.json`)
- Success rate: **0.05** (1/20 episodes).
- Returns (per episode): mean **-11.47**, std 3.69, min -15.30, max 3.99.
- Lengths: mean 49.25, std 3.27, min 35, max 50.
- Env rewards (per step): mean 0.0071, std 0.159, min 0, max 4; fraction env_reward>0 ≈ 0.0020.
- Shaped rewards (per step): mean -0.233, std 0.309, min -0.473, max 8.809.
- Config recorded in metrics: checkpoint `pi0_aloha_sim_tiny_b4`, episodes=20, max_steps=50.

## Qualitative Notes
- Behavior is stable (no physics crashes); mostly low env_reward steps, rare contacts with cube (env_reward>0 only ~0.2% of steps).
- Occasional success observed (1 episode), but overall sparse reward and low success rate indicate the BC policy struggles to complete transfer reliably under this sim setting.

## Stage 1 Baseline (for future RL comparison)
- Task: TransferCube (sim), BC policy (π0 tiny_b4), no RL updates.
- Metrics (20 eps, 50 steps):
  - Success rate: 5%
  - Avg return: -11.47
  - Avg length: 49.25
  - Env reward >0 fraction: ~0.2% of steps
- Use this as the reference when evaluating RL fine-tuning gains.
