"""
environment.py
MDP environment definition for Optimal Irrigation-Fertilization Policy.
State: (soil_moisture, crop_health, weather)
Actions: Do Nothing, Irrigate, Fertilize, Irrigate+Fertilize
"""

import itertools
import numpy as np

# ── State variables ──────────────────────────────────────────────────────────
MOISTURE_LEVELS = ["Low", "Medium", "High"]
HEALTH_LEVELS   = ["Poor", "Fair", "Good"]
WEATHER_LEVELS  = ["Dry", "Normal", "Rainy"]

ACTIONS = ["Do Nothing", "Irrigate", "Fertilize", "Irrigate+Fertilize"]

# ── Build state space ────────────────────────────────────────────────────────
def build_states():
    """Returns list of all 27 state tuples and an index dict."""
    states = list(itertools.product(MOISTURE_LEVELS, HEALTH_LEVELS, WEATHER_LEVELS))
    state_idx = {s: i for i, s in enumerate(states)}
    return states, state_idx

# ── Reward function ──────────────────────────────────────────────────────────
def compute_reward(state, action, water_cost=2.0, fert_cost=1.0):
    """
    R(s, a) — scalar reward for taking action in state.

    Health bonuses: Good=+10, Fair=+4, Poor=-8
    Resource costs: irrigation=-water_cost, fertilizer=-fert_cost
    Overuse penalty: irrigating when moisture=High => extra -3
    Conservation bonus: Do Nothing when conditions already favorable => +2
    """
    moisture, health, weather = state

    # baseline health reward
    health_reward = {"Poor": -8, "Fair": 4, "Good": 10}[health]

    # resource costs
    irrigate  = action in ("Irrigate", "Irrigate+Fertilize")
    fertilize = action in ("Fertilize", "Irrigate+Fertilize")
    cost = 0.0
    if irrigate:
        cost -= water_cost
        if moisture == "High":          # overuse penalty
            cost -= 3.0
    if fertilize:
        cost -= fert_cost

    # conservation bonus
    bonus = 0.0
    if action == "Do Nothing" and moisture != "Low" and health != "Poor":
        bonus += 2.0

    return health_reward + cost + bonus

# ── Transition probabilities ─────────────────────────────────────────────────
def _moisture_transition(moisture, action, weather):
    """
    Returns dict {next_moisture: probability} after applying action in weather.
    """
    irrigate = action in ("Irrigate", "Irrigate+Fertilize")
    probs = {m: 0.0 for m in MOISTURE_LEVELS}

    if irrigate:
        if moisture == "Low":
            probs["Low"]    = 0.10
            probs["Medium"] = 0.85
            probs["High"]   = 0.05
        elif moisture == "Medium":
            probs["Low"]    = 0.05
            probs["Medium"] = 0.50
            probs["High"]   = 0.45
        else:  # High
            probs["Low"]    = 0.00
            probs["Medium"] = 0.20
            probs["High"]   = 0.80
    else:
        # Natural dynamics driven by weather
        if weather == "Rainy":
            if moisture == "Low":
                probs["Low"]    = 0.20
                probs["Medium"] = 0.70
                probs["High"]   = 0.10
            elif moisture == "Medium":
                probs["Low"]    = 0.05
                probs["Medium"] = 0.55
                probs["High"]   = 0.40
            else:
                probs["Low"]    = 0.00
                probs["Medium"] = 0.30
                probs["High"]   = 0.70
        elif weather == "Normal":
            if moisture == "Low":
                probs["Low"]    = 0.65
                probs["Medium"] = 0.30
                probs["High"]   = 0.05
            elif moisture == "Medium":
                probs["Low"]    = 0.20
                probs["Medium"] = 0.65
                probs["High"]   = 0.15
            else:
                probs["Low"]    = 0.05
                probs["Medium"] = 0.40
                probs["High"]   = 0.55
        else:  # Dry
            if moisture == "Low":
                probs["Low"]    = 0.85
                probs["Medium"] = 0.15
                probs["High"]   = 0.00
            elif moisture == "Medium":
                probs["Low"]    = 0.40
                probs["Medium"] = 0.55
                probs["High"]   = 0.05
            else:
                probs["Low"]    = 0.10
                probs["Medium"] = 0.55
                probs["High"]   = 0.35

    # normalize to fix any floating point drift
    total = sum(probs.values())
    return {m: v / total for m, v in probs.items()}


def _health_transition(moisture, health, action):
    """
    Returns dict {next_health: probability}.
    Health improves with adequate moisture and fertilizer; degrades when dry.
    """
    fertilize = action in ("Fertilize", "Irrigate+Fertilize")
    probs = {h: 0.0 for h in HEALTH_LEVELS}

    if moisture == "Low":
        # stress conditions — health tends to decline
        if health == "Good":
            probs["Good"]  = 0.30
            probs["Fair"]  = 0.55
            probs["Poor"]  = 0.15
        elif health == "Fair":
            probs["Good"]  = 0.10
            probs["Fair"]  = 0.45
            probs["Poor"]  = 0.45
        else:  # Poor
            probs["Good"]  = 0.05
            probs["Fair"]  = 0.20
            probs["Poor"]  = 0.75
    elif moisture == "Medium":
        if health == "Good":
            probs["Good"]  = 0.75
            probs["Fair"]  = 0.22
            probs["Poor"]  = 0.03
        elif health == "Fair":
            probs["Good"]  = 0.35
            probs["Fair"]  = 0.55
            probs["Poor"]  = 0.10
        else:
            probs["Good"]  = 0.10
            probs["Fair"]  = 0.45
            probs["Poor"]  = 0.45
    else:  # High moisture (possible overwatering stress)
        if health == "Good":
            probs["Good"]  = 0.65
            probs["Fair"]  = 0.30
            probs["Poor"]  = 0.05
        elif health == "Fair":
            probs["Good"]  = 0.30
            probs["Fair"]  = 0.55
            probs["Poor"]  = 0.15
        else:
            probs["Good"]  = 0.10
            probs["Fair"]  = 0.40
            probs["Poor"]  = 0.50

    # Fertilize boost: shifts probability toward better health
    if fertilize:
        boost = {"Poor": 0.20, "Fair": 0.25, "Good": 0.10}[health]
        if health != "Good":
            next_h = HEALTH_LEVELS[HEALTH_LEVELS.index(health) + 1]
            probs[next_h]  += boost
            probs[health]  -= boost

    total = sum(probs.values())
    return {h: v / total for h, v in probs.items()}


def _weather_transition(weather, rainy_prob=None):
    """
    Simple Markov chain for weather.
    Persists with p=0.60, transitions to each neighbor with p=0.20.
    rainy_prob: optional override for probability of transitioning to Rainy
                (used in the higher-rainfall experiment).
    """
    idx = WEATHER_LEVELS.index(weather)
    n   = len(WEATHER_LEVELS)
    probs = {w: 0.0 for w in WEATHER_LEVELS}

    if rainy_prob is not None:
        # custom: Dry has higher chance of going to Normal/Rainy
        if weather == "Dry":
            probs["Dry"]    = 1.0 - rainy_prob - 0.10
            probs["Normal"] = 0.10
            probs["Rainy"]  = rainy_prob
        else:
            probs[weather]  = 0.60
            left  = WEATHER_LEVELS[(idx - 1) % n]
            right = WEATHER_LEVELS[(idx + 1) % n]
            probs[left]  += 0.20
            probs[right] += 0.20
    else:
        probs[weather] = 0.60
        left  = WEATHER_LEVELS[(idx - 1) % n]
        right = WEATHER_LEVELS[(idx + 1) % n]
        probs[left]  += 0.20
        probs[right] += 0.20

    total = sum(probs.values())
    return {w: max(0.0, v / total) for w, v in probs.items()}


def build_transitions(states, state_idx, rainy_prob=None):
    """
    T[s][a][s2] = P(s2 | s, a)
    Returns nested dict of floats; each T[s][a] sums to 1.0.
    """
    T = {}
    for s in states:
        moisture, health, weather = s
        T[s] = {}
        for action in ACTIONS:
            T[s][action] = {}
            m_probs = _moisture_transition(moisture, action, weather)
            h_probs = _health_transition(moisture, health, action)
            w_probs = _weather_transition(weather, rainy_prob)

            for s2 in states:
                m2, h2, w2 = s2
                T[s][action][s2] = m_probs[m2] * h_probs[h2] * w_probs[w2]

            # explicit normalization guard
            row_sum = sum(T[s][action].values())
            T[s][action] = {s2: v / row_sum for s2, v in T[s][action].items()}
    return T


def build_rewards(states, water_cost=2.0, fert_cost=1.0):
    """R[s][a] scalar reward."""
    return {s: {a: compute_reward(s, a, water_cost, fert_cost) for a in ACTIONS}
            for s in states}


def build_mdp(water_cost=2.0, fert_cost=1.0, rainy_prob=None):
    """Convenience: returns (states, state_idx, T, R)."""
    states, state_idx = build_states()
    T = build_transitions(states, state_idx, rainy_prob)
    R = build_rewards(states, water_cost, fert_cost)
    return states, state_idx, T, R


def validate_transitions(T, states):
    """Check every row sums to 1.0 within tolerance."""
    errors = []
    for s in states:
        for a in ACTIONS:
            row_sum = sum(T[s][a].values())
            if abs(row_sum - 1.0) > 1e-9:
                errors.append(f"T[{s}][{a}] sums to {row_sum}")
    if errors:
        raise ValueError("Transition matrix validation failed:\n" + "\n".join(errors))
    return True
