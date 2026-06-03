# MDP Environments for Disassembly Task Control

This repository contains the OpenAI Gym environments used in the paper:

> **Reinforcement Learning for Disassembly Task Control**  
> Sachini Weerasekara, Wei Li, Jacqueline Isaacs, Sagar Kamarthi  
> *Computers & Industrial Engineering*, Volume 190, April 2024  
> DOI: [10.1016/j.cie.2024.110044](https://doi.org/10.1016/j.cie.2024.110044)  
> ScienceDirect: [https://www.sciencedirect.com/science/article/pii/S0360835224001657](https://www.sciencedirect.com/science/article/pii/S0360835224001657)

---

## Overview

End-of-life product disassembly is a critical step in remanufacturing, but managing disassembly lines under real-world uncertainties — stochastic product arrivals, uncertain component yields, and fluctuating demand — is highly complex. This work formulates disassembly task control as a **Markov Decision Process (MDP)** and trains a **Deep Reinforcement Learning (DRL)** agent using Trust Region Policy Optimization (TRPO) to decide which product type to send to each workstation at each timestep.

The trained DRL policy achieves:
- **21% reduction** in inventory accumulation
- **12% improvement** in demand satisfaction

compared to the Multiple Elman Neural Networks (MENN) baseline method.

---

## Environments

Three environments model disassembly systems for different consumer electronics products, each increasing in complexity.

### Setup A — QLED TV (`DisassemblyTaskControlEnv_SetupA.py`)

| Property | Value |
|---|---|
| Product | QLED (Quantum-dot LED) TV |
| Actions | 7 (6 product routing actions + idle) |
| Observation space | 12-dimensional |
| Workstations | 4 |
| Components tracked | 5 (E, O, L, D, Q) |
| Episode length | 5,000 steps |

### Setup B — OLED TV (`DisassemblyTaskControlEnv_SetupB.py`)

| Property | Value |
|---|---|
| Product | OLED (Organic LED) TV |
| Actions | 10 (9 product routing actions + idle) |
| Observation space | 17-dimensional |
| Workstations | 5 |
| Components tracked | 6 (E, M, O, L, D, Q) |
| Episode length | 1,000 steps |

### Setup C — QD-OLED TV (`DisassemblyTaskControl_SetupC.py`)

| Property | Value |
|---|---|
| Product | QD-OLED (Quantum Dot OLED) TV |
| Actions | 17 (16 product routing actions + idle) |
| Observation space | 29-dimensional |
| Workstations | 7 (each with 2 parallel slots) |
| Components tracked | 9 (B, E, M, O, F, C, Q, L, D) |
| Episode length | 5,000 steps |

---

## Environment Details

### State Space

Each environment's observation vector concatenates three groups:

1. **WIP inventory** — units currently in workstation queues and in-process
2. **Finished component inventory** — units of each recoverable component on hand
3. **Remaining demand** — unfulfilled cumulative demand for each component type

### Action Space

Each discrete action routes one product unit to a specific workstation. The final action (idle) sends nothing. Stochastic Poisson-distributed arrivals bring new products each timestep regardless of the action taken.

### Reward Function

The reward at each step penalises:
- **Inventory accumulation**: weighted sum of all on-hand inventory (`w₀ = −0.5` to `−1.0`)
- **Queue delay**: penalty for routing to a congested workstation when demand is urgent (`w₁ = −0.7` to `−1.0`)

### Stochastic Dynamics

| Source | Distribution |
|---|---|
| Product arrivals | Poisson |
| Component demand | Poisson |
| Processing times | Deterministic (fixed per workstation) |

---

## Installation

```bash
pip install gym numpy stable-baselines3 sb3-contrib
```

---

## Usage

```python
from DisassemblyTaskControlEnv_SetupA import InventoryEnv

env = InventoryEnv()
obs = env.reset()

for _ in range(1000):
    action = env.action_space.sample()   # replace with your policy
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

### Training with TRPO (Setup C example)

```python
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
from DisassemblyTaskControl_SetupC import InventoryEnv, make_env

env = DummyVecEnv([make_env(i) for i in range(10)])
model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5_000_000, log_interval=50)
model.save("trpo_setupC")
```

---

## Repository Structure

```
.
├── DisassemblyTaskControlEnv_SetupA.py   # QLED TV environment
├── DisassemblyTaskControlEnv_SetupB.py   # OLED TV environment
└── DisassemblyTaskControl_SetupC.py      # QD-OLED TV environment (+ training script)
```

---

## Citation

If you use these environments in your research, please cite:

```bibtex
@article{weerasekara2024reinforcement,
  title   = {Reinforcement Learning for Disassembly Task Control},
  author  = {Weerasekara, Sachini and Li, Wei and Isaacs, Jacqueline and Kamarthi, Sagar},
  journal = {Computers \& Industrial Engineering},
  volume  = {190},
  year    = {2024},
  doi     = {10.1016/j.cie.2024.110044}
}
```

---

## License

This code is provided for academic use. Please contact the authors for other uses.
