# Aloha TransferCube RL 微调实验计划（pi0_tiny_b4）

## 1. 研究目标概述
- 在 `gym_aloha/AlohaTransferCube-v0` 上，对 OpenPI（`pi0_tiny_b4`）做 residual RL 微调，重点研究：
  - BC baseline 的成功率和行为特征。
  - contact-based reward + 距离 shaping（cube_dist_lambda）的学习信号影响。
  - 单步前向很慢（~10s+/step）的情况下，RL 的 sample efficiency 与 throughput 问题。

## 2. 环境与模型配置
- 硬件：4× RTX 3090 + E5-2699，JAX/XLA，headless (`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`)，`obs_type=pixels_agent_pos`，三路摄像头（top/left_wrist/right_wrist），state=agent_pos。
- 模型：`pi0_tiny_b4`，配置 `pi0_aloha_sim_tiny_b4`，checkpoint 路径：`checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params`。

## 3. 实验 Stage 划分
- Stage1：BC-only 评估 (`examples/aloha_sim/eval_policy.py`)
  - 命令模板：
    ```bash
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python examples/aloha_sim/eval_policy.py \
      --checkpoint checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params \
      --config-name pi0_aloha_sim_tiny_b4 \
      --episodes 20 \
      --max-episode-steps 50 \
      --save-metrics outputs/tiny_b4_eval_20ep_50steps.json
    ```
  - 已有结果：success ≈ 5%，avg_return ≈ -11.5（20×50 步），可作为 RL 提升对照。

- Stage2：最小 REINFORCE residual 头 (`examples/aloha_sim/rl_train_pi0_tiny.py`)
  - 结构：冻结 BC 采样头，新增 state→action 线性残差 + log_std 高斯，动作 clip[-0.5,0.5]；可 `--disable-residual`。
  - loss：`-mean(adv * logp)`，adv=标准化回报；on-policy REINFORCE，无 baseline/GAE。
  - 采样：base_action + residual 噪声（可调 `--residual-std`）。

- Stage3：profiling + contact shaping
  - Profiling：policy 前向 ~16–22s/step，瓶颈在 `model.sample_actions`；env.step ~0.01s。
  - 多 GPU：可多进程用 `CUDA_VISIBLE_DEVICES` 绑定不同实验；未做 pmap。
  - Contact shaping：env_reward 0..4 → [0,0.2,0.5,1.0,3.0]（`--allow-contact-shaping`）。

- Stage4：cube_pos 暴露 + cube_dist_lambda + reward_components + analyze_rl_log
  - compute_reward 分量：`env_term`（接触映射或原始 env_reward）、`dist_term`（-0.1*||state||）、`act_pen`（-0.01*||action||）、`success_bonus`（+5 on success）、`cube_shaping`（-lambda*||cube_pos-cube_goal||，goal 目前用初始 box pose 占位），均写入 info.reward_components。
  - `rl_train_pi0_tiny.py` 记录 mean_cube_dist / reward 分量均值到 JSON；`analyze_rl_log.py` 可绘制 success/cube_dist 曲线。

## 4. 现阶段关键开放问题
- BC 在新 env wrapper 下的真实 success baseline：`eval_policy` vs `rl_train` BC-only 采集是否一致，需对齐 compute_reward 参数。
- success≈0 的 regime 下，cube 距离是否已有改善：用 `analyze_rl_log.py` 的 mean_cube_dist 曲线验证。
- 单步 ~10s 前向的吞吐：是否需要 batched env、多 GPU 并行或更轻视觉骨干（TODO）。

## 5. 下一步计划（实验建议）
1) 小 sweep：`cube_dist_lambda` ∈ {0.05, 0.1, 0.2}，固定 `residual_std=0.02`，快速配置（ep-per-iter=2, iters=4, steps=50, gamma=0.97），观察 success 与 mean_cube_dist 变化。
   - 命令模板：
   ```bash
   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
   python examples/aloha_sim/rl_train_pi0_tiny.py \
     --checkpoint checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params \
     --config-name pi0_aloha_sim_tiny_b4 \
     --episodes-per-iter 2 \
     --train-iters 4 \
     --max-episode-steps 50 \
     --gamma 0.97 \
     --allow-contact-shaping \
     --cube-dist-lambda {LAMBDA} \
     --residual-std 0.02 \
     --log-json outputs/rl_tiny_cube{LAMBDA}.json
   ```
   - 关注指标：eval/batch success_rate、mean_cube_dist、mean_cube_shaping。

2) residual_std sweep：`residual_std` ∈ {0.0(禁用), 0.01, 0.02, 0.05}，固定 `cube_dist_lambda=0.1`，观察成功率与动作稳定性。
   - 命令同上，替换 `--residual-std`，可选 `--disable-residual` 作为 0.0 对照。

3) 对齐 eval 逻辑：在 `eval_policy` 加 `--allow-contact-shaping/--cube-dist-lambda` 后重新跑 Stage1，验证基线与训练侧一致性；记录新的 success/return 作为对照。

后续如需提升吞吐，可尝试：降低分辨率/单摄像头、批量多 env 前向、或替换更轻模型（TODO）。***
