import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")  # 服务器上无显示时用无头后端
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main(npz_path: pathlib.Path, output: pathlib.Path, dims=(0, 1, 2)) -> None:
    data = np.load(npz_path)
    states = data["state"]  # [T, state_dim]

    i, j, k = dims
    xyz = states[:, [i, j, k]]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linewidth=2)

    ax.set_xlabel(f"state[{i}]")
    ax.set_ylabel(f"state[{j}]")
    ax.set_zlabel(f"state[{k}]")
    ax.set_title(f"3D trajectory from {npz_path.name}")

    # 让坐标轴比例尽量接近 1:1:1
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()
    z_range = xyz[:, 2].max() - xyz[:, 2].min()
    max_range = max(x_range, y_range, z_range)
    if max_range > 0:
        x_mid = (xyz[:, 0].max() + xyz[:, 0].min()) / 2
        y_mid = (xyz[:, 1].max() + xyz[:, 1].min()) / 2
        z_mid = (xyz[:, 2].max() + xyz[:, 2].min()) / 2
        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved 3D trajectory to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=pathlib.Path, help="路径：traj_XXXX.npz")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("traj_3d.png"),
        help="输出图片路径 (默认: traj_3d.png)",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs=3,
        default=(0, 1, 2),
        help="用哪三个维度作为 (x, y, z)，默认 (0,1,2)",
    )
    args = parser.parse_args()

    main(args.npz_path, args.output, tuple(args.dims))
