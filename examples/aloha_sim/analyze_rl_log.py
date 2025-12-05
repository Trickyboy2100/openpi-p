"""
Analyze RL training JSON log produced by rl_train_pi0_tiny.py --log-json.

Expected JSON structure:
{
  "records": [
    {
      "iter": int,
      "batch_success_rate": float,
      "batch_avg_env_r": float,
      "batch_avg_shaped_ret": float,
      "eval_stats": {"success_rate": float, "avg_env_return": float, "avg_shaped_return": float},
      "mean_policy_step_dt": float,
      "mean_env_step_dt": float
    },
    ...
  ]
}

Usage:
  python examples/aloha_sim/analyze_rl_log.py --log-json outputs/rl_tiny_stage3_debug.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RL JSON log from rl_train_pi0_tiny.py.")
    parser.add_argument("--log-json", type=str, required=True, help="Path to JSON log (from --log-json).")
    parser.add_argument("--tag", type=str, default="", help="Optional experiment description to include in plot filenames.")
    args = parser.parse_args()

    log_path = Path(args.log_json)
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    with log_path.open("r") as f:
        data = json.load(f)
    records = data.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("No records found in log.")

    def safe_mean(vals: List[float]) -> float:
        vals = [v for v in vals if not (v is None or (isinstance(v, float) and math.isnan(v)))]
        return float(sum(vals) / len(vals)) if vals else math.nan

    iters = []
    batch_sr = []
    eval_sr = []
    policy_dt = []
    mean_cube_dist = []
    mean_env_term = []
    mean_cube_shaping = []

    for rec in records:
        iters.append(rec.get("iter", len(iters)))
        batch_sr.append(rec.get("batch_success_rate", 0.0))
        eval_sr.append(rec.get("eval_stats", {}).get("success_rate", 0.0))
        policy_dt.append(rec.get("mean_policy_step_dt", 0.0))
        # Optional fields
        mean_cube_dist.append(rec.get("mean_cube_dist", math.nan))
        mean_env_term.append(rec.get("mean_env_term", math.nan))
        mean_cube_shaping.append(rec.get("mean_cube_shaping", math.nan))
        print(
            f"{iters[-1]:4d} | batch_sr={batch_sr[-1]:.3f} | eval_sr={eval_sr[-1]:.3f} | "
            f"mean_cube_dist={mean_cube_dist[-1]:.3f} | "
            f"mean_env_term={mean_env_term[-1]:.3f} | "
            f"mean_cube_shaping={mean_cube_shaping[-1]:.3f}"
        )

    # Create a per-log directory under outputs/plots based on log name and current timestamp.
    import time

    ts = time.strftime("%Y%m%d-%H%M%S")
    base = Path(log_path).stem
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path("outputs/plots") / f"{ts}_{base}{tag_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_curve(y, ylabel, fname):
        plt.figure()
        plt.plot(iters, y, marker="o")
        plt.xlabel("iter")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / fname)
        plt.close()

    save_curve(batch_sr, "batch_success_rate", f"{ts}_batch_success_rate_{base}{tag_suffix}.png")
    save_curve(eval_sr, "eval_success_rate", f"{ts}_eval_success_rate_{base}{tag_suffix}.png")
    save_curve(policy_dt, "mean_policy_step_dt", f"{ts}_mean_policy_step_dt_{base}{tag_suffix}.png")
    # Extra curves if available
    if any(not math.isnan(x) for x in mean_cube_dist):
        save_curve(mean_cube_dist, "mean_cube_dist", f"{ts}_mean_cube_dist_{base}{tag_suffix}.png")
    if any(not math.isnan(x) for x in mean_cube_shaping):
        save_curve(mean_cube_shaping, "mean_cube_shaping", f"{ts}_mean_cube_shaping_{base}{tag_suffix}.png")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
