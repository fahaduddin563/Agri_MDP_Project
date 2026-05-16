import os
import time

import pandas as pd
import streamlit as st

from environment import build_mdp, validate_transitions, ACTIONS, MOISTURE_LEVELS, HEALTH_LEVELS, WEATHER_LEVELS
from solver import value_iteration, policy_iteration
from simulation import run_simulation, run_baseline_simulation
from visualizations import plot_convergence, plot_policy_heatmap, plot_simulation_results, plot_experiments
from main import run_experiment

st.set_page_config(page_title="Agri-MDP AI Demo", page_icon="🌱", layout="wide")

st.title("🌱 Optimal Irrigation–Fertilization Policy")
st.caption("AI Decision Support System using Markov Decision Processes")

with st.sidebar:
    st.header("Demo Controls")
    gamma = st.slider("Discount factor γ", 0.70, 0.99, 0.95, 0.01)
    steps = st.slider("Simulation steps", 100, 1000, 500, 100)
    water_cost = st.slider("Water cost", 1.0, 5.0, 2.0, 0.5)
    fert_cost = st.slider("Fertilizer cost", 1.0, 5.0, 1.0, 0.5)

st.markdown("""
This live demo builds a 27-state agricultural MDP and computes an optimal policy for deciding whether to irrigate, fertilize, apply both, or do nothing.
""")

# Build MDP
states, state_idx, T, R = build_mdp(water_cost=water_cost, fert_cost=fert_cost)
validate_transitions(T, states)

col1, col2, col3, col4 = st.columns(4)
col1.metric("States", len(states))
col2.metric("Actions", len(ACTIONS))
col3.metric("Discount γ", gamma)
col4.metric("Simulation Steps", steps)

st.divider()

run_button = st.button("▶ Run AI Solver", type="primary")

if run_button:
    with st.spinner("Running Value Iteration and Policy Iteration..."):
        start = time.perf_counter()
        V_vi, policy_vi, vi_history = value_iteration(states, T, R, gamma=gamma)
        vi_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        V_pi, policy_pi, pi_iters = policy_iteration(states, state_idx, T, R, gamma=gamma)
        pi_time = (time.perf_counter() - start) * 1000

        policies_match = all(policy_vi[s] == policy_pi[s] for s in states)
        stats_mdp = run_simulation(policy_vi, T, R, n_steps=steps)
        stats_base = run_baseline_simulation(T, R, n_steps=steps)

    st.success("Solver completed successfully.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Value Iteration", f"{len(vi_history)} iterations")
    c2.metric("Policy Iteration", f"{pi_iters} steps")
    c3.metric("VI Runtime", f"{vi_time:.1f} ms")
    c4.metric("Policies Match", "Yes" if policies_match else "No")

    st.subheader("Terminal-Style Result")
    st.code(f"""
============================================================
  Optimal Irrigation–Fertilization Policy
  COMP 569 — MDP Decision Support System
============================================================

Building MDP  (27 states × 4 actions)
States: {len(states)}
Actions: {len(ACTIONS)}
Discount factor γ = {gamma:.2f}

Value Iteration:
Converged in {len(vi_history)} iterations
Final Δ = {vi_history[-1]:.2e}

Policy Iteration:
Converged in {pi_iters} policy improvement steps

{'✓ VI and PI produce identical policies' if policies_match else '⚠ VI and PI policies do not match'}
""", language="text")

    st.subheader("Performance Metrics")
    total = sum(stats_mdp["health_counts"].values())
    good_pct = 100 * stats_mdp["health_counts"]["Good"] / total
    fair_good_pct = 100 * (stats_mdp["health_counts"]["Fair"] + stats_mdp["health_counts"]["Good"]) / total
    do_nothing_pct = 100 * stats_mdp["action_counts"]["Do Nothing"] / total

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Good Health", f"{good_pct:.1f}%")
    m2.metric("Fair or Good", f"{fair_good_pct:.1f}%")
    m3.metric("Do Nothing", f"{do_nothing_pct:.1f}%")
    m4.metric("Avg Reward", f"{stats_mdp['avg_reward']:.2f}")

    st.subheader("Optimal Policy Table")
    policy_rows = []
    for moisture in MOISTURE_LEVELS:
        for health in HEALTH_LEVELS:
            for weather in WEATHER_LEVELS:
                s = (moisture, health, weather)
                policy_rows.append({
                    "Soil Moisture": moisture,
                    "Crop Health": health,
                    "Weather": weather,
                    "Recommended Action": policy_vi[s]
                })
    st.dataframe(pd.DataFrame(policy_rows), use_container_width=True, hide_index=True)

    os.makedirs("figures", exist_ok=True)
    conv_histories = {}
    for g in [0.80, 0.95, 0.99]:
        _, _, hist = value_iteration(states, T, R, gamma=g)
        conv_histories[f"γ = {g:.2f}"] = hist

    plot_convergence(conv_histories, "figures/fig1_convergence.png")
    plot_policy_heatmap(policy_vi, "figures/fig2_policy_heatmap.png")
    plot_simulation_results(stats_mdp, stats_base, "figures/fig3_simulation.png")

    baseline_entry = {
        "label": "Baseline",
        "irrigate_pct": 100 * (stats_mdp["action_counts"]["Irrigate"] + stats_mdp["action_counts"]["Irrigate+Fertilize"]) / total,
        "do_nothing_pct": do_nothing_pct,
        "good_health_pct": good_pct,
        "avg_reward": stats_mdp["avg_reward"],
    }
    experiments = [
        baseline_entry,
        run_experiment("High\nRainfall", rainy_prob=0.40, gamma=gamma, n_steps=steps),
        run_experiment("High\nWater Cost", water_cost=4.0, gamma=gamma, n_steps=steps),
        run_experiment("High\nFert Cost", fert_cost=3.0, gamma=gamma, n_steps=steps),
    ]
    plot_experiments(experiments, "figures/fig4_experiments.png")

    st.subheader("Graphs and Charts")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Convergence", "Policy Heatmap", "Simulation", "Experiments"
    ])
    with tab1:
        st.image("figures/fig1_convergence.png", caption="Value Iteration convergence")
    with tab2:
        st.image("figures/fig2_policy_heatmap.png", caption="Optimal policy heatmap")
    with tab3:
        st.image("figures/fig3_simulation.png", caption="Simulation statistics")
    with tab4:
        st.image("figures/fig4_experiments.png", caption="Experimental comparisons")

    st.subheader("Animated Simulation Preview")
    st.write("The animation below shows the first 20 policy decisions from the simulation trajectory.")
    placeholder = st.empty()
    for i in range(min(20, len(stats_mdp["actions_taken"]))):
        state = stats_mdp["trajectory"][i]
        action = stats_mdp["actions_taken"][i]
        reward = stats_mdp["rewards"][i]
        placeholder.info(
            f"Step {i+1}: Soil Moisture = {state[0]}, Crop Health = {state[1]}, "
            f"Weather = {state[2]} → Action: {action} | Reward: {reward:.1f}"
        )
        time.sleep(0.15)
else:
    st.info("Click **Run AI Solver** to start the live demo.")
