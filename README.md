# Grid Navigation Gym Environment

This repository contains a custom Gymnasium environment `GridEnv` (a configurable MxN grid navigation task) and training/evaluation scripts for PPO and DQN (Stable-Baselines3). It includes utility wrappers, training scripts with TensorBoard logging, and evaluation tools.

---

## Repository structure

```
├── grid_env_gym/
│   └── models/              # Saved model artifacts
│   └── tb_logs/             # TensorBoard logs
│   └── envs.py              # GridEnv implementation
│   └── utils.py             # make_env wrapper (FlattenObservation, TimeLimit, RecordEpisodeStatistics)
│   └── train_ppo.py         # PPO training script ( TB, final model save)
│   └── train_dqn.py         # DQN training script (TB, final model save)
│   └── evaluate.py          # Evaluation script that outputs DataFrame results
├── tests/                   # Test and debugging helpers 
├── requirements.txt         # Library requirements file 
├── README.md                # Project readme (this file)
└── Design_Analysis.pdf      # Design analysis (environment, reward, training choices)
```

---

# Setup

## 1. Requirements

- Python 3.11+ recommended
- Install dependencies (pip):

```bash
python -m pip install -U pip
python -m pip install "stable-baselines3[extra]" gymnasium numpy pandas matplotlib
```

This will install PyTorch (as required by SB3), Gymnasium, and other utilities.

## 2. Optional: create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt  # if you created one
```

---

# How to train

## PPO (recommended for experiments)

Quick smoke test (fast):

```bash
python train_ppo.py --size small # 20k timesteps
```

For a larger training run:

```bash
python train_ppo.py --size medium   # 100k timesteps
python train_ppo.py --size large    # 500k timesteps
```

Models are saved to `models/ppo_grid_*.zip` and TensorBoard logs to `tb_logs/`.

## DQN (off-policy)

Quick smoke test:

```bash
python train_dqn.py --size small
```

Longer run:

```bash
python train_dqn.py --size medium
python train_dqn.py --size large
```

Models are saved to `models/dqn_grid_*.zip` and TensorBoard logs to `tb_logs/`.

---

# How to evaluate

Use the evaluation script which produces a DataFrame with average steps, average reward, and success rate.

```bash
python evaluate.py --ppo-path models/ppo_grid_large.zip --dqn-path models/dqn_grid_large.zip --n-episodes 100
```

The script prints a small table with results. TensorBoard can be launched with:

```bash
tensorboard --logdir tb_logs
```

---

# Notes

- Ensure `utils.make_env` is used both for training and evaluation to avoid wrapper mismatches.

---


