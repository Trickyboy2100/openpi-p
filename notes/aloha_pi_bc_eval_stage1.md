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

## Stage 2/3 Progress (2025-12-04)
- Files touched:
  - `examples/aloha_sim/rl_train_pi0_tiny.py`: residual REINFORCE head, JSON logging, profiling, contact-shaping flag, advantage normalization, Stage3 debug preset.
  - `examples/aloha_sim/rl_loop.py`: reward shaping map, device debug prints, image shape fixes.
  - `examples/aloha_sim/profile_policy.py`: per-stage profiling + device debug.
  - `src/openpi/models/pi0.py`: extra image channel sanity (clip to 3 channels).
- Profiling (single forward, tiny_b4, NHWC 224x224):
  - sample_actions_dt ≈ 15–22s; inputs/tokenize/obs_build ~<0.2s. Bottleneck = model.sample_actions.
- Reward shaping:
  - Optional contact mapping 0..4 → [0, 0.2, 0.5, 1.0, 3.0] (enable via `--allow-contact-shaping`, default on).
  - Advantage used in update: adv = (R - mean) / (std + 1e-6).
- Debug presets:
  - `--stage3-debug` sets episodes-per-iter=2, train-iters=5, max-episode-steps=20, gamma=0.97. Use for quick tiny experiments.
  - JSON log via `--log-json` appends per-iter stats to `{"records": [...]}`.
- Device sanity:
  - Added `[jax-debug] devices=...` and sample_actions/base_policy tensor device prints in policy/profile/train scripts to confirm GPU usage.

## Commands (Stage3 tiny debug example)
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/aloha_sim/rl_train_pi0_tiny.py \
  --checkpoint checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params \
  --config-name pi0_aloha_sim_tiny_b4 \
  --stage3-debug \
  --allow-contact-shaping \
  --log-json outputs/rl_tiny_stage3_debug.json
```
Expect logs with `[jax-debug] devices=...`, per-iter train/eval stats, profiling line, and JSON records (batch_success_rate, eval success_rate, mean_policy_step_dt, etc.). Forward pass remains slow (~16–22s/step); success likely sparse unless reward signal improves.

## Suggested command sequence (runnable, with purposes)
1) **Profile policy forward (per-stage timing + device check)**  
   Purpose: confirm images/device shape and measure inputs/tokenize/obs_build/sample_actions timings.  
   ```bash
   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
   python examples/aloha_sim/profile_policy.py \
     --checkpoint checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params \
     --config-name pi0_aloha_sim_tiny_b4 \
     --iters 6 --warmup 1
   ```  
   Watch `[jax-debug] devices=...` and the timing summary; sample_actions is the bottleneck.

2) **BC-only eval (Stage1 baseline re-run)**  
   Purpose: reproduce baseline success rate/return on tiny_b4.  
   ```bash
   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
   python examples/aloha_sim/eval_policy.py \
     --checkpoint checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params \
     --config-name pi0_aloha_sim_tiny_b4 \
     --episodes 20 \
     --max-episode-steps 50 \
     --save-metrics outputs/tiny_b4_eval_20ep_50steps.json
   ```

3) **Analyze eval metrics (Stage1)**  
   Purpose: summarize success/return stats, optional plots.  
   ```bash
   python examples/aloha_sim/analyze_eval_metrics.py \
     --metrics outputs/tiny_b4_eval_20ep_50steps.json \
     --plot
   ```

4) **Stage2/3 tiny debug RL run**  
   Purpose: quick residual REINFORCE sanity check with contact shaping and JSON logging.  
   ```bash
   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
   python examples/aloha_sim/rl_train_pi0_tiny.py \
     --checkpoint checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params \
     --config-name pi0_aloha_sim_tiny_b4 \
     --stage3-debug \
     --allow-contact-shaping \
     --log-json outputs/rl_tiny_stage3_debug.json
   ```  
   Check console `[train]/[eval]/[profile]` lines and JSON `records` for `batch_success_rate` / `eval success_rate`.

# How to onboard a new Codex session

1. cd ~/Gitclones/openpi
2. source .venv/bin/activate
3. Ask Codex to:
   - read examples/aloha_sim/env.py, rl_loop.py, eval_policy.py, analyze_eval_metrics.py, rl_train_pi0_tiny.py
   - read notes/aloha_pi_bc_eval_stage1.md
4. Then continue Stage 2 RL experiments.
