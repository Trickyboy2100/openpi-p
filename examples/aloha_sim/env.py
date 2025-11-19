import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class AlohaSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation."""

    # original 初始化 AlohaSimEnvironment 类，设置任务名称、观测类型和随机种子
    # def __init__(self, task: str, obs_type: str = "pixels_agent_pos", seed: int = 0) -> None:
    #     np.random.seed(seed)
    #     self._rng = np.random.default_rng(seed)
    #
    #     self._gym = gymnasium.make(task, obs_type=obs_type)
    #
    #     self._last_obs = None
    #     self._done = True
    #     self._episode_reward = 0.0
    
    def __init__(
        self,
        task: str,
        obs_type: str = "pixels_agent_pos",
        seed: int = 0,
        max_episode_steps: int = 400,  # ← 新增：每个 episode 的最大步数
    ) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        self._gym = gymnasium.make(
            task,
            obs_type=obs_type,
            max_episode_steps=max_episode_steps,  # ← 关键
    )

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0



    @override
    def reset(self) -> None:
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False
        self._episode_reward = 0.0

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    # 返回当前回合的累计奖励的最大值
    @override
    def apply_action(self, action: dict) -> None:
        # 将 OpenPI 的动作格式转换为 gym_aloha 所需的格式并执行动作     
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = truncated # or terminated
        self._episode_reward = max(self._episode_reward, reward)

    def _convert_observation(self, gym_obs: dict) -> dict:
        # 从 gym_aloha 的观测构造 OpenPI 所需要的观测格式
        # 这里假设 gym_obs 包含 "pixels" 和 "agent_pos" 键
        img = gym_obs["pixels"]["top"]
        # Resize and convert image to uint8
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        # Convert axis order from [H, W, C] --> [C, H, W]
        # [C, H, W]分别代表 通道、高度、宽度
        # 两者不统一的原因是 PyTorch 和 TensorFlow 对图像数据的默认格式不同
        img = np.transpose(img, (2, 0, 1))

        # 返回包含状态和图像的观测字典
        # python字典的键值对顺序是无序的
        # 这里我们返回的字典包含两个键："state" 和 "images"
        return {
            "state": gym_obs["agent_pos"],
            "images": {"cam_high": img},
        }
