# Setup

This file describes the local development setup for TextToTorque.

The local environment is used for:

- editing code in VS Code
- launching Modal jobs
- logging experiments with Weights & Biases
- running analysis scripts
- making plots and tables

The actual MuJoCo / Unitree G1 training environment runs on Modal or another Linux GPU machine, not locally on macOS.

## 1. Create the Conda Environment

```bash
conda create -n text-to-torque python=3.11 -y
conda activate text-to-torque
```

## 2. Install Local Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install modal wandb numpy pandas matplotlib scipy pyyaml tqdm rich ipython jupyter black ruff pytest
```

## 3. Verify the Environment

```bash
python --version
python -m pip --version
python -c "import modal, wandb, numpy, pandas, matplotlib, yaml, tqdm; print('local env good')"
```

## 4. Set Up Modal

```bash
modal setup
```

To test that Modal is installed locally:

```bash
modal --version
```

## 5. Set Up Weights & Biases

```bash
wandb login
```

To test that W&B is installed:

```bash
wandb --version
```

## 6. Project Structure

```text
external/      External repos and dependencies
scripts/       Project scripts and diagnostics
configs/       Experiment configs
experiments/   Per-run notes and metadata
results/       Figures, tables, and small result files
motions/       Generated or processed motion files
```

Large files such as checkpoints, logs, rollout videos, and generated motion arrays should not be committed to git.

## 7. Notes for macOS

This local environment is not meant to run full humanoid PPO training. macOS is used for development and experiment management.

Training should be run on Modal or another Linux machine with an NVIDIA GPU.

## 8. Useful Commands

Activate the environment:

```bash
conda activate text-to-torque
```

Check that Python and pip are using the conda environment:

```bash
which python
which pip
python -m pip --version
```

Commit setup changes:

```bash
git add setup.md
git commit -m "Add setup instructions"
git push
```