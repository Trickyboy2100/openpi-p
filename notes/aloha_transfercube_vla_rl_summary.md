# Aloha TransferCube VLA + RL 现状备忘录

Timestamp: 2025-12-05T12:40:00+08:00

## 1. Environment & Reward
- 任务：`gym_aloha/AlohaTransferCube-v0`。obs：三路像素（top/左右腕）+ `agent_pos` (14-dof)。env.reward 离散 0..4（接触/抬起/左手触/成功），`info["is_success"]` 表示成功。env wrapper (`examples/aloha_sim/env.py`) 额外暴露 `cube_pos`（env_state 前 7 维）和 `cube_goal_pos`（当前用初始 box pose 占位），并在 done 条件中合并 terminated/truncated/is_success。
- reward 逻辑（`compute_reward`）：env_term（可映射 0..4→[0,0.2,0.5,1.0,3.0]）、dist_term（-0.1‖state‖）、act_pen（-0.01‖a‖）、success_bonus（+5）、可选 cube_shaping（-lambda * ‖cube_pos - cube_goal‖）。

## 2. BC baseline（Pi0 tiny b4, 20×50）对比
| 指标 | tiny_b4_eval_20ep_50steps | tiny_b4_eval_post_envchange |
| --- | --- | --- |
| Episodes | 20 | 20 |
| Success rate | 0.05 | 0.00 |
| Avg return | -11.47 | -12.49 |
| Avg length | 49.25 | 50.0 |
| Env_reward>0 fraction | 0.0020 | 0.0030 |
说明：5% vs 0% 很可能是统计波动/环境细节差异（一次成功/无成功），整体仍然极低。

## 3. RL fine-tuning 尝试（`rl_train_pi0_tiny.py`）
- 架构：冻结 BC 采样头，残差头 state→action 线性 + log_std 高斯，可 `--disable-residual`，loss = -mean(adv*logp)，adv 为标准化回报。奖励可选 contact/cube shaping。
- 典型超参（Stage3 debug）：episodes-per-iter=2, train-iters=5, max-episode-steps=20, gamma=0.97, residual_std=0.02, cube_dist_lambda=0（可调）。
- RL 日志示例（`outputs/rl_tiny_stage3_debug.json`，iters 0-4）：batch_success_rate 全 0，eval_success_rate 全 0，mean_policy_step_dt ≈10.7–11.1s，未见成功率提升。

## 4. Profiling & Practical Limits
- `profile_policy.py`：sample_actions_dt ≈10–11s/step；env step ≈0.015s/step。瓶颈完全在模型前向（`model.sample_actions`）。这意味着在线 RL 每步成本极高，迭代数据量受限。

## 5. Conclusion & Next Steps
- 结论：在当前设定（高延迟前向、稀疏接触奖励）下，VLA+小残差 REINFORCE 未产生可见成功率提升；奖励 shaping 尚未体现成学习进展。
- 下一步：验证任务可学性，建议先做 state-only RL baseline（去掉像素、用低维 obs），或大幅简化视觉/动作空间；同时对齐 eval 与 train 的 reward 参数，观察 cube 距离 shaping 是否带来下降趋势。***
