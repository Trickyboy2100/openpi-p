# scripts/tiny_forward_sanity.py

import pathlib

import jax
import jax.numpy as jnp

from openpi.training.config import get_config
from openpi.training import data_loader
from openpi.models import model as _model


def main():
    # 1. 加载你自定义的 tiny 训练配置
    #    名字就是你在 training/config.py 里注册的那个
    cfg = get_config("pi0_aloha_sim_tiny_b4")

    # 2. 创建一个只取 1 个 batch 的 dataloader
    #    skip_norm_stats=True：这里只是做前向 sanity check，不重新算 norm stats
    loader = data_loader.create_data_loader(
        cfg,
        num_batches=1,
        skip_norm_stats=True,
    )
    batch = next(iter(loader))
    print("Got batch type:", type(batch))

    # openpi 的 dataloader 返回的是 (Observation, Actions)
    obs, act = batch
    print("Obs type:", type(obs))
    print("Act type:", type(act))

    # 3. 用 openpi 自带的 helper 从 Orbax checkpoint 里恢复参数
    ckpt_path = pathlib.Path(
        "checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params"
    )
    print("Restoring params from:", ckpt_path)

    params = _model.restore_params(
        ckpt_path,
        dtype=jnp.bfloat16,  # 如果你训练是 fp32，可以改成 jnp.float32
    )

    # 4. 用 config + params 创建带权重的模型
    #    重点：这里是 cfg.model.load(...)，而不是 model.init/apply
    model = cfg.model.load(params)
    print("Model created:", type(model))

    # 5. 前向一次：算个 loss 当 sanity check
    key = jax.random.key(0)
    loss = model.compute_loss(key, obs, act)
    print("Loss shape:", loss.shape)

    # 如果想顺便看 sample_actions 也行，打开下面注释：
    # actions_out = model.sample_actions(key, obs)
    # print("Sampled actions shape:", actions_out.shape)


if __name__ == "__main__":
    main()
