"""
Stage 1 tooling: analyze saved eval metrics for BC policies on AlohaTransferCube-v0.

Metrics are produced by examples/aloha_sim/eval_policy.py when called with --save-metrics.

Example:
  python examples/aloha_sim/eval_policy.py \
    --checkpoint checkpoints/pi0_aloha_sim_tiny_b4/aloha_sim_tiny_gpu_b4/199/params \
    --config-name pi0_aloha_sim_tiny_b4 \
    --episodes 20 \
    --max-episode-steps 200 \
    --save-metrics outputs/tiny_b4_transfercube_eval_20ep.json

  python examples/aloha_sim/analyze_eval_metrics.py \
    --metrics outputs/tiny_b4_transfercube_eval_20ep.json \
    --plot
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import time

import numpy as np


def summarize_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics from eval metrics."""
    returns = np.asarray(data.get("returns", []), dtype=np.float64)
    lengths = np.asarray(data.get("lengths", []), dtype=np.float64)
    successes = np.asarray(data.get("successes", []), dtype=bool)

    # flatten per-step rewards
    env_rewards = [r for ep in data.get("env_rewards", []) for r in ep]
    shaped_rewards = [r for ep in data.get("shaped_rewards", []) for r in ep]
    env_rewards = np.asarray(env_rewards, dtype=np.float64) if env_rewards else np.array([], dtype=np.float64)
    shaped_rewards = np.asarray(shaped_rewards, dtype=np.float64) if shaped_rewards else np.array([], dtype=np.float64)

    summary: Dict[str, Any] = {}
    summary["episodes"] = len(returns)
    summary["success_rate"] = float(successes.mean()) if len(successes) else 0.0

    def stats(arr: np.ndarray) -> Dict[str, float]:
        if arr.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    summary["return_stats"] = stats(returns)
    summary["length_stats"] = stats(lengths)
    summary["env_reward_stats"] = stats(env_rewards)
    summary["env_reward_fraction_pos"] = float((env_rewards > 0).mean()) if env_rewards.size else 0.0
    summary["shaped_reward_stats"] = stats(shaped_rewards)

    summary["config"] = data.get("config", {})
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print summary dict."""
    print("---- Eval Summary ----")
    print(f"Episodes: {summary.get('episodes', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0.0):.3f}")
    rstats = summary.get("return_stats", {})
    print(
        f"Returns: mean={rstats.get('mean', 0.0):.3f}, std={rstats.get('std', 0.0):.3f}, "
        f"min={rstats.get('min', 0.0):.3f}, max={rstats.get('max', 0.0):.3f}"
    )
    lstats = summary.get("length_stats", {})
    print(
        f"Lengths: mean={lstats.get('mean', 0.0):.2f}, std={lstats.get('std', 0.0):.2f}, "
        f"min={lstats.get('min', 0.0):.2f}, max={lstats.get('max', 0.0):.2f}"
    )
    env_stats = summary.get("env_reward_stats", {})
    print(
        f"Env rewards: mean={env_stats.get('mean', 0.0):.3f}, std={env_stats.get('std', 0.0):.3f}, "
        f"min={env_stats.get('min', 0.0):.3f}, max={env_stats.get('max', 0.0):.3f}, "
        f"frac_pos={summary.get('env_reward_fraction_pos', 0.0):.3f}"
    )
    shaped_stats = summary.get("shaped_reward_stats", {})
    print(
        f"Shaped rewards: mean={shaped_stats.get('mean', 0.0):.3f}, std={shaped_stats.get('std', 0.0):.3f}, "
        f"min={shaped_stats.get('min', 0.0):.3f}, max={shaped_stats.get('max', 0.0):.3f}"
    )
    if "config" in summary:
        print(f"Config: {summary['config']}")


def plot_metrics(data: Dict[str, Any], out_dir: Path) -> None:
    """Save basic histograms for returns/env_rewards/shaped_rewards."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = time.strftime("%Y%m%d-%H%M%S")
    base = Path(data.get("config", {}).get("checkpoint", "metrics")).stem
    out_dir = out_dir / f"{ts}_{base}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_hist(arr_list, title, fname, bins=30):
        arr = np.asarray(arr_list, dtype=np.float64)
        if arr.size == 0:
            return
        plt.figure()
        plt.hist(arr, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
        plt.title(title)
        plt.xlabel("value")
        plt.ylabel("count")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / fname)
        plt.close()

    save_hist(data.get("returns", []), "Per-episode returns", f"{ts}_returns_hist_{base}.png")
    # flatten per-step rewards
    env_rewards = [r for ep in data.get("env_rewards", []) for r in ep]
    shaped_rewards = [r for ep in data.get("shaped_rewards", []) for r in ep]
    save_hist(env_rewards, "Per-step env rewards", f"{ts}_env_rewards_hist_{base}.png")
    save_hist(shaped_rewards, "Per-step shaped rewards", f"{ts}_shaped_rewards_hist_{base}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze eval metrics JSON produced by eval_policy.py.")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics JSON (from eval_policy.py --save-metrics).")
    parser.add_argument("--plot", action="store_true", help="If set, generate PNG plots in outputs/plots.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with metrics_path.open("r") as f:
        data = json.load(f)

    summary = summarize_metrics(data)
    print_summary(summary)

    if args.plot:
        out_dir = Path("outputs/plots")
        plot_metrics(data, out_dir)
        print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
