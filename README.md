# Optimal Irrigation–Fertilization Policy
### COMP 569 – Artificial Intelligence | Project 3
**Paramveer Singh** | California State University Channel Islands

---

## Overview

An MDP-based decision support system for sustainable farm management.  
The agent learns the optimal policy — **when to irrigate, fertilize, both, or do nothing** —  
under uncertain weather, using Value Iteration and Policy Iteration.

- **State space:** 27 states (3 moisture × 3 health × 3 weather levels)  
- **Action space:** Do Nothing, Irrigate, Fertilize, Irrigate+Fertilize  
- **Solvers:** Value Iteration (47 iters) and Policy Iteration (6 iters) → identical policy  
- **Results:** 68% Good crop health, 91% Fair-or-Good, 41% Do Nothing (resource-efficient)

---

## Project Structure

```
agri_mdp/
├── environment.py       MDP definition (states, actions, T, R)
├── solver.py            Value Iteration + Policy Iteration
├── simulation.py        Monte Carlo rollouts & statistics
├── visualizations.py    All 4 figures (matplotlib)
├── main.py              Entry point — runs everything
├── figures/             Generated figures (created on first run)
└── README.md
```

---

## Setup

**Requirements:** Python 3.8+, NumPy, Matplotlib

```bash
pip install numpy matplotlib
```

---

## Running the Project

### Full run (all solvers, all experiments, all figures)
```bash
cd agri_mdp
python main.py
```

### Value Iteration only
```bash
python main.py --solver vi
```

### Policy Iteration only
```bash
python main.py --solver pi
```

### Custom discount factor
```bash
python main.py --gamma 0.99
```

### Longer simulation (1000 steps)
```bash
python main.py --steps 1000
```

### Interactive demo — query the policy state-by-state
```bash
python main.py --demo
```

### Skip figure generation (faster runs)
```bash
python main.py --no-figs
```

---

## Example Output

```
============================================================
  Optimal Irrigation–Fertilization Policy
  COMP 569 — MDP Decision Support System
============================================================

──────────────────────────────────────────────────────────
  Building MDP  (27 states × 4 actions)
──────────────────────────────────────────────────────────
  States:  27
  Actions: 4  →  ['Do Nothing', 'Irrigate', 'Fertilize', 'Irrigate+Fertilize']
  Transitions validated ✓
  Discount factor γ = 0.95

──────────────────────────────────────────────────────────
  Value Iteration
──────────────────────────────────────────────────────────
  Converged in 47 iterations  (3.2 ms)
  Final Δ = 5.81e-07

──────────────────────────────────────────────────────────
  Policy Iteration
──────────────────────────────────────────────────────────
  Converged in 6 policy improvement steps  (1.1 ms)

  ✓ VI and PI produce identical policies
```

### Sample Policy Table

| Moisture | Health | Weather | Recommended Action |
|----------|--------|---------|--------------------|
| Low      | Poor   | Dry     | Irrigate+Fertilize |
| Low      | Fair   | Dry     | Irrigate           |
| Medium   | Poor   | Normal  | Fertilize          |
| Medium   | Fair   | Normal  | Do Nothing         |
| High     | Good   | Rainy   | Do Nothing         |

---

## Generated Figures

| File | Description |
|------|-------------|
| `figures/fig1_convergence.png` | VI convergence curves for γ ∈ {0.80, 0.95, 0.99} |
| `figures/fig2_policy_heatmap.png` | Policy heatmap: all 27 states × 3 weather panels |
| `figures/fig3_simulation.png` | Simulation: health dist., action pie, running avg reward |
| `figures/fig4_experiments.png` | Experiment comparisons (rainfall, water cost, fert cost) |

---

## Key Results

| Metric | Value |
|--------|-------|
| Time in Good health | 68.4% |
| Time in Fair/Good health | 91.0% |
| Do Nothing (resource savings) | 41.2% |
| Avg reward/step (MDP) | +5.83 |
| Avg reward/step (Naive baseline) | +1.24 |

---

## AI Disclosure

This project was developed with assistance from Claude (Anthropic) for code structure,  
debugging, and report writing. All AI concepts, algorithm logic, and experimental  
analysis represent the author's own understanding of the course material.
