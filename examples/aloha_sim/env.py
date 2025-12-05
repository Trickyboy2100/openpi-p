import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
from gym_aloha.env import BOX_POSE  # access sampled cube init pose for goal placeholder


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
        self._cube_goal_pos = None  # placeholder for cube goal (from BOX_POSE start pose)

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
        # Capture goal pose from BOX_POSE sampled during gym_aloha reset (pos+quat, take position)
        if BOX_POSE[0] is not None:
            self._cube_goal_pos = np.asarray(BOX_POSE[0])[:3]
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
    def apply_action(self, action: dict) -> tuple[float, dict]:
        # 将 OpenPI 的动作格式转换为 gym_aloha 所需的格式并执行动作
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        # gym_aloha marks success via reward==4 and sets terminated; truncated is time-limit.
        is_success = info.get("is_success", False) if info else False
        self._done = bool(truncated) or bool(terminated) or bool(is_success)
        # 累积奖励，便于 RL 统计；如果更合适取 max，可自行调整
        self._episode_reward += reward
        info = info or {}
        if self._cube_goal_pos is not None:
            info["cube_goal_pos"] = self._cube_goal_pos
        return reward, info

    def _convert_observation(self, gym_obs: dict) -> dict:
        # 从 gym_aloha 的观测构造 OpenPI 所需要的观测格式
        # 这里假设 gym_obs 包含 "pixels" 和 "agent_pos" 键
        img = gym_obs["pixels"]["top"]
        # Resize and convert image to uint8
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        # Ensure HWC layout for downstream policy (if channel-first, transpose)
        if img.ndim == 3 and img.shape[-1] != 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim != 3 or img.shape[-1] not in (1, 3):
            raise ValueError(f"Unexpected image shape after resize: {img.shape}")
        cube_pos = None
        # gym_aloha TransferCube exposes env_state = qpos[16:], where the first 7 entries are box pose (pos+quat).
        if "env_state" in gym_obs:
            env_state = np.asarray(gym_obs["env_state"])
            if env_state.size >= 7:
                cube_pos = env_state[:7]
        # 保持 HWC 格式，后续 convert_env_obs 期望 HWC
        obs_out = {
            "state": gym_obs["agent_pos"],
            "images": {"cam_high": img},
        }
        if cube_pos is not None:
            obs_out["cube_pos"] = cube_pos
        if self._cube_goal_pos is not None:
            obs_out["cube_goal_pos"] = self._cube_goal_pos
        return obs_out
