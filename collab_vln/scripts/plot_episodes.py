#!/usr/bin/env python3
"""Plot episode start/goal locations on a 2D map for manual review.

Usage:
  python collab_vln/scripts/plot_episodes.py \
      --input collab_vln/episodes/town10hd_templates.json \
      --output collab_vln/episodes/town10hd_map.png
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def plot_episodes(episodes: list[dict], out_path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(14, 12))

    # Collect all points to determine bounds
    all_x, all_y = [], []
    for ep in episodes:
        all_x.extend([ep["ugv_spawn"]["x"], ep["goal"]["x"], ep["uav_spawn"]["x"]])
        all_y.extend([ep["ugv_spawn"]["y"], ep["goal"]["y"], ep["uav_spawn"]["y"]])

    margin = 30
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m) — CARLA world coordinate")
    ax.set_ylabel("Y (m) — CARLA world coordinate")
    ax.set_title(title or "Episode Map Overview")

    for i, ep in enumerate(episodes):
        color = COLORS[i % len(COLORS)]
        sx, sy = ep["ugv_spawn"]["x"], ep["ugv_spawn"]["y"]
        gx, gy = ep["goal"]["x"], ep["goal"]["y"]
        ux, uy = ep["uav_spawn"]["x"], ep["uav_spawn"]["y"]
        did = ep["id"]
        dist = ep["distance_m"]

        # UGV start (square)
        ax.plot(sx, sy, marker="s", color=color, markersize=10, zorder=3)
        # Goal (star)
        ax.plot(gx, gy, marker="*", color=color, markersize=14, zorder=3)
        # Arrow from start to goal
        ax.annotate("", xy=(gx, gy), xytext=(sx, sy),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.6))
        # UAV spawn (triangle)
        ax.plot(ux, uy, marker="^", color=color, markersize=8, zorder=2, alpha=0.6)

        # Label: episode id + distance
        mid_x, mid_y = (sx + gx) / 2, (sy + gy) / 2
        offset_x, offset_y = 5, 5
        # Alternate label position
        if i % 3 == 0:
            offset_y = -8
        elif i % 3 == 2:
            offset_x = -15

        ax.annotate(f"{did}\n{dist:.0f}m", (mid_x, mid_y),
                    xytext=(mid_x + offset_x, mid_y + offset_y),
                    fontsize=7, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85))

    # Per-episode detail table below the main plot
    legend_lines = [
        f"{ep['id']}  |  #{ep['ugv_spawn']['index']:>3d} → goal ({ep['goal']['x']:6.0f}, {ep['goal']['y']:6.0f})  |  {ep['distance_m']:6.1f}m"
        f"  |  UGV yaw: {ep['ugv_spawn']['yaw']:6.1f}°"
        for ep in episodes
    ]

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor="none", label="■  UGV start (spawn point)"),
        mpatches.Patch(facecolor="none", edgecolor="none", label="★  Goal (target location)"),
        mpatches.Patch(facecolor="none", edgecolor="none", label="▲  UAV hover position"),
        mpatches.Patch(facecolor="none", edgecolor="none", label="→  Navigation path (straight-line)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

    # Detail text below
    detail_text = "\n".join(legend_lines[:15])  # show first 15
    fig.text(0.5, 0.01, detail_text, ha="center", va="bottom",
             fontsize=6, fontfamily="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot episodes on 2D map")
    parser.add_argument("--input", required=True, help="Path to episode templates JSON")
    parser.add_argument("--output", default=None, help="Output PNG path (default: <input>.png)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    episodes = data["episodes"]
    map_name = data["meta"]["map"]

    out_path = Path(args.output) if args.output else in_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_episodes(episodes, out_path, title=f"{map_name} — {len(episodes)} Episodes")


if __name__ == "__main__":
    main()
