import gymnasium as gym
import numpy as np

def main():
    # 这个 task 名字和 openpi 里默认的一致（你可以对照 main.py 里的 args.task）
    env_id = "gym_aloha/AlohaTransferCube-v0"
    env = gym.make(env_id, obs_type="pixels_agent_pos")

    print("Action space:", env.action_space)
    print("Observation space keys:", env.reset(seed=0)[0].keys())

    obs, info = env.reset(seed=0)
    print("Initial obs keys:", obs.keys())
    print("agent_pos shape:", obs["agent_pos"].shape)
    print("pixels[top] shape:", obs["pixels"]["top"].shape)

    episode_reward = 0.0
    for t in range(10):
        # 随机动作演示：后面会被你的 RL policy 替换
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += float(reward)

        print(f"step {t}: reward={reward}, done={done}")
        print("  agent_pos shape:", obs["agent_pos"].shape)
        print("  pixels[top] shape:", obs["pixels"]["top"].shape)

        if done:
            print("Episode finished, total_reward =", episode_reward)
            break

if __name__ == "__main__":
    main()
