#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from openpi.training.config import get_config
from openpi.training import data_loader
from openpi.training import optimizer
import orbax.checkpoint as ocp

# 加载配置
cfg = get_config("pi0_aloha_sim_tiny_b4")  # 或 tiny 单卡配置

# 构建 mesh（单机多卡 / 单卡都适用）
mesh = jax.sharding.Mesh(jax.devices(), ("B",))

# 创建数据 loader
dl, _ = data_loader.create_data_loader(cfg, mesh)
batch = next(iter(dl))

# 构建模型
model = cfg.model.create(jax.random.key(0))
variables = model.init(batch)  # 仅构图

# 加载 checkpoint 参数
ckpt_path = "checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params"
ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
params = ckptr.restore(ckpt_path)

# 前向运行
out = model.apply({"params": params}, batch)

# 打印结果
print(out.shape if hasattr(out, "shape") else type(out))

