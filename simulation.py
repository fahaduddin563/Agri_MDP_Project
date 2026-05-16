"""
simulation.py
Monte Carlo simulation using a computed MDP policy.
"""

import random
from environment import ACTIONS, MOISTURE_LEVELS, HEALTH_LEVELS, WEATHER_LEVELS


def sample_next_state(state, action, T):
    """Sample next state from the transition distribution T[state][action]."""
    states   = list(T[state][action].keys())
    probs    = list(T[state][action].values())

    # Weighted random choice
    r = random.random()
    cumulative = 0.0
    for s, p in zip(states, probs):
        cumulative += p
        if r <= cumulative:
            return s
    return states[-1]   # fallback for floating-point edge


def run_simulation(policy, T, R, n_steps=500, initial_state=None, seed=42):
    """
    Run a Monte Carlo episode under the given policy.

    Returns
    -------
    stats : dict with keys:
        trajectory          list of states visited
        actions_taken       list of actions taken
        rewards             list of per-step rewards
        health_counts       dict health -> count
        action_counts       dict action -> count
        avg_reward          float
    """
    random.seed(seed)

    all_states = list(T.keys())
    if initial_state is None:
        state = random.choice(all_states)
    else:
        state = initial_state

    trajectory    = [state]
    actions_taken = []
    rewards       = []
    health_counts = {h: 0 for h in HEALTH_LEVELS}
    action_counts = {a: 0 for a in ACTIONS}

    for _ in range(n_steps):
        action = policy[state]
        reward = R[state][action]

        actions_taken.append(action)
        rewards.append(reward)
        health_counts[state[1]] += 1
        action_counts[action]   += 1

        state = sample_next_state(state, action, T)
        trajectory.append(state)

    return {
        "trajectory":    trajectory,
        "actions_taken": actions_taken,
        "rewards":       rewards,
        "health_counts": health_counts,
        "action_counts": action_counts,
        "avg_reward":    sum(rewards) / len(rewards),
    }


def run_baseline_simulation(T, R, n_steps=500, seed=42):
    """
    Naive baseline: always apply Irrigate+Fertilize regardless of state.
    """
    naive_policy = {s: "Irrigate+Fertilize" for s in T.keys()}
    return run_simulation(naive_policy, T, R, n_steps=n_steps, seed=seed)


def print_stats(stats, label="Simulation"):
    total = sum(stats["health_counts"].values())
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Total steps:         {total}")
    print(f"  Average reward/step: {stats['avg_reward']:.3f}")
    print(f"\n  Crop Health Distribution:")
    for h in HEALTH_LEVELS:
        pct = 100 * stats["health_counts"][h] / total
        print(f"    {h:6s}: {stats['health_counts'][h]:4d}  ({pct:.1f}%)")
    print(f"\n  Action Distribution:")
    for a in ACTIONS:
        pct = 100 * stats["action_counts"][a] / total
        print(f"    {a:22s}: {stats['action_counts'][a]:4d}  ({pct:.1f}%)")
    print(f"{'='*55}\n")
