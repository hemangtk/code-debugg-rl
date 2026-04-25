"""Generate the six required training/evaluation plots."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_reward(rows: List[Dict], out: str) -> None:
    steps = [r["step"] for r in rows]
    reward = [r["reward"] for r in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, reward, color="tab:blue", linewidth=1, alpha=0.5, label="per-episode reward")
    if len(reward) >= 10:
        win = 10
        smooth = [
            sum(reward[max(0, i - win + 1) : i + 1]) / min(win, i + 1)
            for i in range(len(reward))
        ]
        plt.plot(steps, smooth, color="tab:blue", linewidth=2.5, label=f"rolling mean (w={win})")
    plt.title("Total reward over training episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_accuracy(rows: List[Dict], out: str) -> None:
    steps = [r["step"] for r in rows]
    acc = [r["rolling_accuracy"] for r in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, acc, color="tab:green", linewidth=2, label="rolling accuracy (w=20)")
    plt.title("Fix accuracy over training")
    plt.xlabel("Episode")
    plt.ylabel("All tests pass (%)")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_cave_rate(rows: List[Dict], out: str) -> None:
    steps = [r["step"] for r in rows]
    cave = [r["rolling_cave_rate"] for r in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, cave, color="tab:red", linewidth=2, label="rolling cave rate (w=20)")
    plt.title("Cave rate over training (lower is better)")
    plt.xlabel("Episode")
    plt.ylabel("Caved (%)")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_investigation(rows: List[Dict], out: str) -> None:
    steps = [r["step"] for r in rows]
    depth = [r["tool_calls"] for r in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, depth, color="tab:orange", linewidth=1, alpha=0.5, label="tool calls / episode")
    if len(depth) >= 10:
        win = 10
        smooth = [
            sum(depth[max(0, i - win + 1) : i + 1]) / min(win, i + 1)
            for i in range(len(depth))
        ]
        plt.plot(steps, smooth, color="tab:orange", linewidth=2.5, label=f"rolling mean (w={win})")
    plt.title("Investigation depth over training")
    plt.xlabel("Episode")
    plt.ylabel("Tool calls per episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_comparison(results_path: str, out: str) -> None:
    with open(results_path) as f:
        results = json.load(f)
    metrics = ["accuracy", "cave_rate", "avg_investigation_depth", "resistance_rate"]
    labels = ["Fix accuracy", "Cave rate", "Avg tool calls", "Resistance rate"]
    base = [results["base"][m] for m in metrics]
    trained = [results["trained"][m] for m in metrics]

    x = list(range(len(metrics)))
    width = 0.35
    plt.figure(figsize=(9, 4.5))
    plt.bar([xi - width / 2 for xi in x], base, width, label="Base model", color="tab:gray")
    plt.bar([xi + width / 2 for xi in x], trained, width, label="Trained model", color="tab:blue")
    plt.xticks(x, labels)
    plt.title(f"Base vs trained model (n={results['n']})")
    plt.ylabel("Value")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_difficulty(rows: List[Dict], out: str) -> None:
    steps = [r["step"] for r in rows]
    axes = ["bug_subtlety", "misconception_convincingness", "code_complexity", "correct_user_ratio"]
    plt.figure(figsize=(8, 4.5))
    for axis in axes:
        vals = [r["difficulty"].get(axis, 0.0) for r in rows]
        plt.plot(steps, vals, linewidth=2, label=axis)
    fc = [r["difficulty"].get("followup_count", 0) / 3.0 for r in rows]
    plt.plot(steps, fc, linewidth=2, linestyle="--", label="followup_count / 3")
    plt.title("Curriculum difficulty axes over training")
    plt.xlabel("Episode")
    plt.ylabel("Difficulty (0..1)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-log", default="training_log.jsonl")
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = _load_jsonl(args.training_log) if os.path.exists(args.training_log) else []

    if rows:
        plot_reward(rows, os.path.join(args.out_dir, "1_reward.png"))
        plot_accuracy(rows, os.path.join(args.out_dir, "2_fix_accuracy.png"))
        plot_cave_rate(rows, os.path.join(args.out_dir, "3_cave_rate.png"))
        plot_investigation(rows, os.path.join(args.out_dir, "4_investigation_depth.png"))
        plot_difficulty(rows, os.path.join(args.out_dir, "6_difficulty_axes.png"))
        print(f"Wrote training plots from {len(rows)} episodes.")

    if os.path.exists(args.results):
        plot_comparison(args.results, os.path.join(args.out_dir, "5_base_vs_trained.png"))
        print("Wrote comparison plot.")


if __name__ == "__main__":
    main()
