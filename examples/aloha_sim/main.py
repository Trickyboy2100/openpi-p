import dataclasses
import logging
import pathlib

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro

# 定义命令行参数的数据类, 包含环境配置、服务器连接信息等
@dataclasses.dataclass
# args指代命令行参数
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")

    task: str = "gym_aloha/AlohaTransferCube-v0"
#    task: str = "gym_aloha/AlohaInsertion-v0"

    # 随机种子, 用于环境的随机性控制
    seed: int = 0

    # 规划的动作时间步数，即每次从策略获取的动作序列长度
    #action_horizon: int = 10
    action_horizon: int = 20

    host: str = "0.0.0.0"
    port: int = 8000

    display: bool = False


    # 新增
    max_episode_steps: int = 400
    num_episodes: int = 1      # 如果想一次跑多条视频可以用


def main(args: Args) -> None:
    runtime = _runtime.Runtime(
        environment=_env.AlohaSimEnvironment(
            task=args.task,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,  # 让命令行参数生效
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
        ],
        max_hz=50,
        num_episodes=args.num_episodes,
    )

    runtime.run()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
