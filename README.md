# ICU Sepsis Reinforcement Learning

Reinforcement learning project comparing tabular and deep RL methods for ICU sepsis treatment policy learning on the published ICU-Sepsis benchmark (Choudhary et al., 2024).

## Overview

ICU sepsis treatment is a sequential decision problem: clinicians adjust IV fluids and vasopressors as patient state evolves, with survival revealed only at the end. We model this as a finite MDP and compare three RL agents.

## Methods

- **Q-Learning** (off-policy, tabular)
- **SARSA** (on-policy, tabular)
- **DQN** (deep RL extension with two-layer MLP)

## Environment

- 716 discrete patient states (K-means clusters of 47 clinical features)
- 25 discrete actions (5×5 grid of IV fluid × vasopressor doses)
- Sparse terminal reward: +1 survival, 0 death
- Built from ~17,000 MIMIC-III sepsis patient records

## Results

| Method     | Final Return (Mean ± SE) | Seeds |
|------------|--------------------------|-------|
| Random     | 0.78 (baseline)          | —     |
| Q-Learning | 0.7856 ± 0.0009          | 10    |
| SARSA      | 0.7880 ± 0.0007          | 10    |
| DQN        | 0.7913 ± 0.0035          | 3     |
| Optimal    | 0.88 (upper bound)       | —     |

Key findings:
- SARSA produced the most stable tabular policy, reducing seed-to-seed variance by 22% vs Q-Learning
- DQN achieved the highest final return but with limited gain relative to added complexity
- All three methods cleared the random baseline of 0.78

## Tech Stack

Python · PyTorch · NumPy · Matplotlib · icu_sepsis

## Files

- `Group11.ipynb` — full training, tuning, and evaluation pipeline
- `requirements.txt` — dependencies

Johns Hopkins Carey | MS Business Analytics & AI
