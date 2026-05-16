"""
solver.py
Value Iteration and Policy Iteration solvers for a finite MDP.
"""

import numpy as np
from environment import ACTIONS


# ── Value Iteration ──────────────────────────────────────────────────────────

def value_iteration(states, T, R, gamma=0.95, tol=1e-6):
    """
    Solves the MDP via Value Iteration.

    Returns
    -------
    V       : dict  state -> optimal value
    policy  : dict  state -> optimal action string
    history : list  max-delta per iteration (for convergence plot)
    """
    V = {s: 0.0 for s in states}
    policy = {}
    history = []

    iteration = 0
    while True:
        delta = 0.0
        for s in states:
            q_values = {}
            for a in ACTIONS:
                q = R[s][a] + gamma * sum(T[s][a][s2] * V[s2] for s2 in states)
                q_values[a] = q

            best_action = max(q_values, key=q_values.get)
            best_val    = q_values[best_action]
            delta = max(delta, abs(best_val - V[s]))
            V[s]       = best_val
            policy[s]  = best_action

        history.append(delta)
        iteration += 1

        if delta < tol:
            break

    return V, policy, history


# ── Policy Iteration ─────────────────────────────────────────────────────────

def policy_evaluation(states, state_idx, T, R, policy, gamma=0.95, tol=1e-8):
    """
    Exact policy evaluation by solving the linear system:
        V^π = R^π + γ T^π V^π
    Returns V^π as a dict {state -> value}.
    """
    n = len(states)

    # Build R_pi vector and T_pi matrix
    R_pi = np.array([R[s][policy[s]] for s in states], dtype=float)
    T_pi = np.zeros((n, n), dtype=float)
    for i, s in enumerate(states):
        for j, s2 in enumerate(states):
            T_pi[i, j] = T[s][policy[s]][s2]

    # Solve: (I - γ T_pi) V = R_pi
    A = np.eye(n) - gamma * T_pi
    v_vec = np.linalg.solve(A, R_pi)

    return {s: v_vec[state_idx[s]] for s in states}


def policy_iteration(states, state_idx, T, R, gamma=0.95):
    """
    Solves the MDP via Policy Iteration.

    Returns
    -------
    V           : dict  state -> optimal value
    policy      : dict  state -> optimal action string
    pi_history  : int   number of policy improvement steps
    """
    # initialise with arbitrary policy
    policy = {s: ACTIONS[0] for s in states}
    pi_history = 0

    while True:
        # Policy evaluation
        V = policy_evaluation(states, state_idx, T, R, policy, gamma)

        # Policy improvement
        policy_stable = True
        for s in states:
            q_values = {
                a: R[s][a] + gamma * sum(T[s][a][s2] * V[s2] for s2 in states)
                for a in ACTIONS
            }
            best_action = max(q_values, key=q_values.get)
            if best_action != policy[s]:
                policy[s]     = best_action
                policy_stable = False

        pi_history += 1
        if policy_stable:
            break

    return V, policy, pi_history
