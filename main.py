"""
main.py
Entry point for the MDP-based Agricultural Decision Support System.

Usage
-----
  python main.py                         # full run with all figures
  python main.py --solver vi             # Value Iteration only
  python main.py --solver pi             # Policy Iteration only
  python main.py --gamma 0.99            # custom discount factor
  python main.py --steps 1000           # longer simulation
  python main.py --demo                  # short interactive demo
"""

import argparse
import sys
import time

from environment import (
    build_mdp, validate_transitions,
    MOISTURE_LEVELS, HEALTH_LEVELS, WEATHER_LEVELS, ACTIONS
)
from solver import value_iteration, policy_iteration
from simulation import run_simulation, run_baseline_simulation, print_stats
from visualizations import (
    plot_convergence, plot_policy_heatmap,
    plot_simulation_results, plot_experiments
)


# ── Pretty-print helpers ─────────────────────────────────────────────────────

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def print_policy_table(policy):
    header = f"{'Moisture':10s}  {'Health':8s}  {'Weather':8s}  {'Recommended Action'}"
    print(header)
    print("─" * len(header))
    for moisture in MOISTURE_LEVELS:
        for health in HEALTH_LEVELS:
            for weather in WEATHER_LEVELS:
                s = (moisture, health, weather)
                a = policy.get(s, "?")
                print(f"  {moisture:8s}  {health:8s}  {weather:8s}  {a}")


# ── Experiment helpers ───────────────────────────────────────────────────────

def run_experiment(label, water_cost=2.0, fert_cost=1.0,
                   rainy_prob=None, gamma=0.95, n_steps=500, seed=42):
    states, state_idx, T, R = build_mdp(water_cost, fert_cost, rainy_prob)
    V, policy, history = value_iteration(states, T, R, gamma=gamma)
    stats = run_simulation(policy, T, R, n_steps=n_steps, seed=seed)

    total = sum(stats["action_counts"].values())
    irrigate_pct = 100 * (stats["action_counts"]["Irrigate"] +
                          stats["action_counts"]["Irrigate+Fertilize"]) / total
    dn_pct    = 100 * stats["action_counts"]["Do Nothing"] / total
    good_pct  = 100 * stats["health_counts"]["Good"] / total
    avg_r     = stats["avg_reward"]

    return {
        "label":            label,
        "irrigate_pct":     irrigate_pct,
        "do_nothing_pct":   dn_pct,
        "good_health_pct":  good_pct,
        "avg_reward":       avg_r,
        "stats":            stats,
    }


# ── Convergence sweep ────────────────────────────────────────────────────────

def gamma_sweep(states, T, R):
    section("Discount Factor Sweep  (Value Iteration vs Policy Iteration)")
    gammas = [0.70, 0.80, 0.90, 0.95, 0.99]
    from solver import policy_iteration
    from environment import build_states
    _, state_idx = build_states()
    print(f"  {'γ':>5}  {'VI iters':>10}  {'PI iters':>10}  {'Avg Reward':>12}")
    print("  " + "─"*46)
    for g in gammas:
        _, _, vi_hist = value_iteration(states, T, R, gamma=g)
        _, _, pi_iters = policy_iteration(states, state_idx, T, R, gamma=g)
        _, policy_g, _ = value_iteration(states, T, R, gamma=g)
        sim = run_simulation(policy_g, T, R, n_steps=500)
        print(f"  {g:>5.2f}  {len(vi_hist):>10d}  {pi_iters:>10d}  {sim['avg_reward']:>12.3f}")
    return gammas, {g: value_iteration(states, T, R, gamma=g)[2] for g in [0.80, 0.95, 0.99]}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agri-MDP Decision Support System")
    parser.add_argument("--solver",  choices=["vi", "pi", "both"], default="both")
    parser.add_argument("--gamma",   type=float, default=0.95)
    parser.add_argument("--steps",   type=int,   default=500)
    parser.add_argument("--demo",    action="store_true", help="Interactive demo mode")
    parser.add_argument("--no-figs", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Optimal Irrigation–Fertilization Policy")
    print("  COMP 569 — MDP Decision Support System")
    print("="*60)

    # ── Build baseline MDP ──────────────────────────────────────────────────
    section("Building MDP  (27 states × 4 actions)")
    states, state_idx, T, R = build_mdp()
    validate_transitions(T, states)
    print(f"  States:  {len(states)}")
    print(f"  Actions: {len(ACTIONS)}  →  {ACTIONS}")
    print(f"  Transitions validated ✓")
    print(f"  Discount factor γ = {args.gamma}")

    # ── Solve ───────────────────────────────────────────────────────────────
    vi_history = None

    if args.solver in ("vi", "both"):
        section("Value Iteration")
        t0 = time.perf_counter()
        V_vi, policy_vi, vi_history = value_iteration(states, T, R, gamma=args.gamma)
        elapsed = time.perf_counter() - t0
        print(f"  Converged in {len(vi_history)} iterations  ({elapsed*1000:.1f} ms)")
        print(f"  Final Δ = {vi_history[-1]:.2e}")

    if args.solver in ("pi", "both"):
        section("Policy Iteration")
        t0 = time.perf_counter()
        V_pi, policy_pi, pi_iters = policy_iteration(states, state_idx, T, R, gamma=args.gamma)
        elapsed = time.perf_counter() - t0
        print(f"  Converged in {pi_iters} policy improvement steps  ({elapsed*1000:.1f} ms)")

    # Use VI policy as primary (both are identical)
    policy = policy_vi if args.solver in ("vi", "both") else policy_pi

    # ── Verify policies match ────────────────────────────────────────────────
    if args.solver == "both":
        mismatches = [s for s in states if policy_vi[s] != policy_pi[s]]
        if mismatches:
            print(f"\n  ⚠ Policy mismatch in {len(mismatches)} states!")
        else:
            print(f"\n  ✓ VI and PI produce identical policies")

    # ── Policy table ─────────────────────────────────────────────────────────
    section("Optimal Policy  (all 27 states)")
    print_policy_table(policy)

    # ── Baseline simulation ──────────────────────────────────────────────────
    section("Simulation  (MDP Policy vs Naive Baseline)")
    stats_mdp      = run_simulation(policy, T, R, n_steps=args.steps)
    stats_baseline = run_baseline_simulation(T, R, n_steps=args.steps)
    print_stats(stats_mdp,      label=f"MDP Policy  ({args.steps} steps)")
    print_stats(stats_baseline, label=f"Naive Baseline  ({args.steps} steps)")
    print(f"  Reward improvement: {stats_mdp['avg_reward'] - stats_baseline['avg_reward']:+.2f} per step")

    # ── Gamma sweep ──────────────────────────────────────────────────────────
    gammas, conv_histories = gamma_sweep(states, T, R)

    # ── Experiments ──────────────────────────────────────────────────────────
    section("Experiments")
    exp_results = []

    print("  [1/3] Higher Rainfall Probability (p_rain → 0.40)...")
    e1 = run_experiment("High\nRainfall", rainy_prob=0.40)
    exp_results.append(e1)
    print(f"        Avg reward: {e1['avg_reward']:.3f}")

    print("  [2/3] Higher Water Cost (irrigation cost → 4.0)...")
    e2 = run_experiment("High\nWater Cost", water_cost=4.0)
    exp_results.append(e2)
    print(f"        Avg reward: {e2['avg_reward']:.3f}")

    print("  [3/3] Higher Fertilizer Penalty (fert cost → 3.0)...")
    e3 = run_experiment("High\nFert Cost", fert_cost=3.0)
    exp_results.append(e3)
    print(f"        Avg reward: {e3['avg_reward']:.3f}")

    # Prepend baseline for comparison
    total = sum(stats_mdp["action_counts"].values())
    baseline_entry = {
        "label":           "Baseline",
        "irrigate_pct":    100*(stats_mdp["action_counts"]["Irrigate"] +
                                stats_mdp["action_counts"]["Irrigate+Fertilize"]) / total,
        "do_nothing_pct":  100*stats_mdp["action_counts"]["Do Nothing"] / total,
        "good_health_pct": 100*stats_mdp["health_counts"]["Good"] / total,
        "avg_reward":      stats_mdp["avg_reward"],
    }
    all_experiments = [baseline_entry] + exp_results

    # ── Figures ──────────────────────────────────────────────────────────────
    if not args.no_figs:
        section("Generating Figures  →  figures/")

        print("  [1/4] Convergence curve...")
        convergence_data = {f"γ = {g}": conv_histories[g] for g in [0.80, 0.95, 0.99]}
        plot_convergence(convergence_data)

        print("  [2/4] Policy heatmap...")
        plot_policy_heatmap(policy)

        print("  [3/4] Simulation results...")
        plot_simulation_results(stats_mdp, stats_baseline)

        print("  [4/4] Experiment comparison...")
        plot_experiments(all_experiments)

        print("\n  All figures saved to ./figures/")

    # ── Interactive demo ─────────────────────────────────────────────────────
    if args.demo:
        section("Interactive Demo — Query the Policy")
        print("  Enter farm conditions to get the recommended action.")
        print("  (Press Ctrl+C to exit)\n")

        m_opts = {str(i+1): m for i, m in enumerate(MOISTURE_LEVELS)}
        h_opts = {str(i+1): h for i, h in enumerate(HEALTH_LEVELS)}
        w_opts = {str(i+1): w for i, w in enumerate(WEATHER_LEVELS)}

        try:
            while True:
                print("  Soil Moisture:  1=Low  2=Medium  3=High")
                m = m_opts.get(input("  → ").strip(), "Medium")
                print("  Crop Health:    1=Poor  2=Fair  3=Good")
                h = h_opts.get(input("  → ").strip(), "Fair")
                print("  Weather:        1=Dry  2=Normal  3=Rainy")
                w = w_opts.get(input("  → ").strip(), "Normal")

                s = (m, h, w)
                a = policy.get(s, "?")
                v = V_vi.get(s, 0.0)
                print(f"\n  ╔══════════════════════════════════════╗")
                print(f"  ║  State:  {m:7s} | {h:4s} | {w:6s}       ║")
                print(f"  ║  Action: {a:28s} ║")
                print(f"  ║  Value:  {v:+.3f}                        ║")
                print(f"  ╚══════════════════════════════════════╝\n")
        except KeyboardInterrupt:
            print("\n  Demo ended.")

    print("\n" + "="*60)
    print("  Run complete.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
