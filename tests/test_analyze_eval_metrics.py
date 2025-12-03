import numpy as np

from examples.aloha_sim.analyze_eval_metrics import summarize_metrics


def test_summarize_metrics_basic():
    data = {
        "returns": [1.0, 2.0],
        "lengths": [10, 20],
        "successes": [True, False],
        "env_rewards": [[0, 1], [2]],
        "shaped_rewards": [[-1, -2], [-3]],
        "config": {"checkpoint": "ckpt", "episodes": 2, "max_steps": 5},
    }
    summary = summarize_metrics(data)

    assert summary["episodes"] == 2
    assert np.isclose(summary["success_rate"], 0.5)
    rstats = summary["return_stats"]
    assert np.isclose(rstats["mean"], 1.5)
    assert np.isclose(rstats["min"], 1.0)
    assert np.isclose(rstats["max"], 2.0)

    env_stats = summary["env_reward_stats"]
    assert np.isclose(env_stats["mean"], np.array([0, 1, 2]).mean())
    assert np.isclose(summary["env_reward_fraction_pos"], 2 / 3)

    shaped_stats = summary["shaped_reward_stats"]
    assert np.isclose(shaped_stats["min"], -3.0)
    assert np.isclose(shaped_stats["max"], -1.0)
