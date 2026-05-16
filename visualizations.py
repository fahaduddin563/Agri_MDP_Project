"""
visualizations.py
Generate all four project figures:
  Fig 1  — Value Iteration convergence curve
  Fig 2  — Policy heatmap across all 27 states
  Fig 3  — Monte Carlo simulation results
  Fig 4  — Comparative experiment bar charts
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from environment import MOISTURE_LEVELS, HEALTH_LEVELS, WEATHER_LEVELS, ACTIONS

# ── Colour palette ───────────────────────────────────────────────────────────
COLOURS = {
    "Do Nothing":          "#4C9BE8",   # blue
    "Irrigate":            "#3DBF7F",   # green
    "Fertilize":           "#F5A623",   # amber
    "Irrigate+Fertilize":  "#E85454",   # red
}
ACTION_LABELS = ["Do Nothing", "Irrigate", "Fertilize", "Irrigate+Fertilize"]
ACTION_SHORT  = ["Do\nNothing", "Irrigate", "Fertilize", "Irr+\nFert"]

os.makedirs("figures", exist_ok=True)


# ── Figure 1: Convergence curve ──────────────────────────────────────────────

def plot_convergence(history_dict, save_path="figures/fig1_convergence.png"):
    """
    history_dict: {label: [delta per iteration]}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    line_styles = ["-", "--", "-."]
    palette     = ["#2E75B6", "#E85454", "#3DBF7F"]

    for i, (label, history) in enumerate(history_dict.items()):
        ax.plot(range(1, len(history) + 1), history,
                linestyle=line_styles[i % 3],
                color=palette[i % 3],
                linewidth=2.2, label=label)

    ax.axhline(y=1e-6, color="gray", linestyle=":", linewidth=1.2, label="Threshold (1×10⁻⁶)")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=13, fontweight="bold")
    ax.set_ylabel("Max ΔV  (log scale)", fontsize=13, fontweight="bold")
    ax.set_title("Figure 1 — Value Iteration Convergence", fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_facecolor("#F9FBFD")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ── Figure 2: Policy heatmap ─────────────────────────────────────────────────

def plot_policy_heatmap(policy, save_path="figures/fig2_policy_heatmap.png"):
    """
    3 panels (one per weather), rows=crop health, cols=soil moisture.
    Each cell coloured by recommended action.
    """
    action_to_int = {a: i for i, a in enumerate(ACTION_LABELS)}
    cmap = LinearSegmentedColormap.from_list(
        "policy", list(COLOURS.values()), N=4)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Figure 2 — Optimal Policy Heatmap (all 27 states)",
                 fontsize=14, fontweight="bold", y=1.02)

    for col, weather in enumerate(WEATHER_LEVELS):
        ax = axes[col]
        matrix = np.zeros((len(HEALTH_LEVELS), len(MOISTURE_LEVELS)))
        annot  = []
        for r, health in enumerate(reversed(HEALTH_LEVELS)):
            row_ann = []
            for c, moisture in enumerate(MOISTURE_LEVELS):
                s = (moisture, health, weather)
                a = policy.get(s, "Do Nothing")
                matrix[r, c] = action_to_int[a]
                row_ann.append(a.replace("Irrigate+Fertilize", "Irr+Fert"))
            annot.append(row_ann)

        im = ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=3.5,
                       aspect="auto", interpolation="nearest")

        for r in range(len(HEALTH_LEVELS)):
            for c in range(len(MOISTURE_LEVELS)):
                ax.text(c, r, annot[r][c], ha="center", va="center",
                        fontsize=10.5, fontweight="bold", color="white",
                        bbox=dict(facecolor="none", edgecolor="none"))

        ax.set_xticks(range(len(MOISTURE_LEVELS)))
        ax.set_xticklabels(MOISTURE_LEVELS, fontsize=11)
        ax.set_yticks(range(len(HEALTH_LEVELS)))
        ax.set_yticklabels(list(reversed(HEALTH_LEVELS)), fontsize=11)
        ax.set_xlabel("Soil Moisture", fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Crop Health", fontsize=11, fontweight="bold")
        ax.set_title(f"Weather: {weather}", fontsize=12, fontweight="bold",
                     color="#1F3864")
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # legend
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in COLOURS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=10.5, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ── Figure 3: Simulation results ─────────────────────────────────────────────

def plot_simulation_results(stats, baseline_stats,
                             save_path="figures/fig3_simulation.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 3 — Monte Carlo Simulation Results (500 steps)",
                 fontsize=14, fontweight="bold")

    # — 3a: Health distribution (MDP vs baseline) —
    ax = axes[0]
    x  = np.arange(len(HEALTH_LEVELS))
    w  = 0.35
    total = sum(stats["health_counts"].values())
    mdp_vals = [100 * stats["health_counts"][h] / total     for h in HEALTH_LEVELS]
    bas_vals = [100 * baseline_stats["health_counts"][h] / total for h in HEALTH_LEVELS]
    bars1 = ax.bar(x - w/2, mdp_vals, w, color="#2E75B6", label="MDP Policy",   zorder=3)
    bars2 = ax.bar(x + w/2, bas_vals, w, color="#E85454", label="Naive Baseline", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(HEALTH_LEVELS, fontsize=12)
    ax.set_ylabel("% of Time Steps", fontsize=11)
    ax.set_title("Crop Health Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_facecolor("#F9FBFD")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    # — 3b: Action breakdown (pie) —
    ax = axes[1]
    ac = stats["action_counts"]
    vals   = [ac[a] for a in ACTION_LABELS]
    colors = [COLOURS[a] for a in ACTION_LABELS]
    wedge_props = dict(width=0.55, edgecolor="white", linewidth=2)
    ax.pie(vals, labels=None, colors=colors, autopct="%1.1f%%",
           startangle=140, pctdistance=0.75,
           wedgeprops=wedge_props, textprops={"fontsize": 10})
    ax.set_title("Action Breakdown (MDP)", fontsize=12, fontweight="bold")
    patches = [mpatches.Patch(color=COLOURS[a], label=a) for a in ACTION_LABELS]
    ax.legend(handles=patches, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=2)

    # — 3c: Cumulative reward over time —
    ax = axes[2]
    rewards = stats["rewards"]
    cum_r   = np.cumsum(rewards)
    steps   = np.arange(1, len(rewards) + 1)
    ax.plot(steps, cum_r / steps, color="#2E75B6", linewidth=2.2, label="MDP Policy")
    ax.axhline(y=baseline_stats["avg_reward"], color="#E85454",
               linestyle="--", linewidth=1.8, label=f"Baseline avg ({baseline_stats['avg_reward']:.2f})")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Running Avg Reward/Step", fontsize=11)
    ax.set_title("Running Average Reward", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(linestyle="--", alpha=0.4)
    ax.set_facecolor("#F9FBFD")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ── Figure 4: Comparative experiments ───────────────────────────────────────

def plot_experiments(experiment_results, save_path="figures/fig4_experiments.png"):
    """
    experiment_results: list of dicts, each with keys:
        label, irrigate_pct, do_nothing_pct, good_health_pct, avg_reward
    """
    labels      = [e["label"] for e in experiment_results]
    irr_pcts    = [e["irrigate_pct"]   for e in experiment_results]
    dn_pcts     = [e["do_nothing_pct"] for e in experiment_results]
    good_pcts   = [e["good_health_pct"] for e in experiment_results]
    avg_rewards = [e["avg_reward"]     for e in experiment_results]

    x = np.arange(len(labels))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Figure 4 — Comparative Experimental Analysis",
                 fontsize=14, fontweight="bold")

    # — Left: action mix —
    ax = axes[0]
    b1 = ax.bar(x - w, irr_pcts,  w, color="#3DBF7F", label="Irrigate %",    zorder=3)
    b2 = ax.bar(x,     dn_pcts,   w, color="#4C9BE8", label="Do Nothing %",  zorder=3)
    b3 = ax.bar(x + w, good_pcts, w, color="#F5A623", label="Good Health %", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5, rotation=12, ha="right")
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Action Mix & Health Outcomes", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_facecolor("#F9FBFD"); ax.set_ylim(0, 90)
    for bar in [b1, b2, b3]:
        for rect in bar:
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_height() + 0.8,
                    f"{rect.get_height():.1f}", ha="center", va="bottom", fontsize=8.5)

    # — Right: average reward —
    ax = axes[1]
    bar_colors = ["#2E75B6", "#3DBF7F", "#F5A623", "#E85454"]
    bars = ax.bar(x, avg_rewards, 0.5, color=bar_colors[:len(labels)], zorder=3,
                  edgecolor="white", linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5, rotation=12, ha="right")
    ax.set_ylabel("Avg. Reward per Step", fontsize=11)
    ax.set_title("Average Reward per Step", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_facecolor("#F9FBFD"); ax.set_ylim(0, max(avg_rewards) * 1.2)
    for rect in bars:
        ax.text(rect.get_x() + rect.get_width()/2,
                rect.get_height() + 0.05,
                f"{rect.get_height():.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")
