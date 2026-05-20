# TextToTorque

RL tracking of text-generated humanoid motions for Unitree G1 in MuJoCo.

This is a CS224R final project studying how text-generated reference motions can be converted into physically executable humanoid robot policies through reinforcement learning.

## Overview

Recent text-to-motion models can generate diverse humanoid motions from natural language prompts, but these motions are not guaranteed to be physically feasible for a torque-controlled robot. This project studies the gap between kinematic motion generation and reinforcement-learned physical control.

We use a MuJoCo-based Unitree G1 tracking pipeline to investigate which properties of generated reference motions make them easier or harder for PPO policies to learn.

## Research Questions

- Which properties of generated reference motions predict downstream PPO tracking success?
- What failure modes occur when plausible-looking generated motions are used as physical tracking targets?
- Can curriculum-style termination improve learning on difficult generated motions?

## Approach

We generate text-specified humanoid reference motions, train reinforcement learning policies to track them in simulation, and analyze how motion-level diagnostics relate to tracking performance. We compare baseline PPO tracking with curriculum-based training interventions designed to improve stability and sample efficiency.

## Repository Structure

```text
external/      External repos and dependencies
scripts/       Project scripts and diagnostics
configs/       Experiment configs
experiments/   Per-run notes and metadata
results/       Figures, tables, and videos
motions/       Generated or processed motion files
```

## Local Setup

The Mac/local environment is for editing code, launching Modal jobs, and analyzing results. KimoLab generation and PPO training should run on Modal or another Linux NVIDIA GPU machine.

Create and activate the local environment:

```bash
cd "/Users/benji/Documents/CS224R Project/text-to-torque"

conda create -n text-to-torque python=3.11 -y
conda activate text-to-torque

python -m pip install --upgrade pip setuptools wheel
python -m pip install modal wandb numpy pandas matplotlib scipy pyyaml tqdm rich ipython jupyter black ruff pytest
```

Load local environment variables:

```bash
set -a
source .env
set +a
```

Verify the local install:

```bash
python -c "import modal, wandb, numpy; print('local env good')"
modal --version
```

Set up Modal if needed:

```bash
modal setup
```

## KimoLab Modal Bring-Up

Fill in `HF_TOKEN=` in `.env`. The token must have access to Meta-Llama-3-8B-Instruct on Hugging Face.

Create the Modal secret:

```bash
set -a
source .env
set +a

modal secret create huggingface HF_TOKEN="$HF_TOKEN"
```

Run the default reference-motion bring-up:

```bash
modal run modal_kimolab.py
```

Run a tiny PPO debug job:

```bash
modal run modal_kimolab.py --train --num-envs 128 --max-iterations 20 --record-train-video
```

For a fresh terminal session:

```bash
conda activate text-to-torque
cd "/Users/benji/Documents/CS224R Project/text-to-torque"
set -a; source .env; set +a
```
