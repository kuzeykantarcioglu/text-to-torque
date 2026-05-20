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
external/      External repos for local inspection only
scripts/       Project scripts and diagnostics
configs/       Experiment configs
experiments/   Per-run notes and metadata
results/       Figures, tables, and small result files
motions/       Generated or processed motion files
```

Large files such as checkpoints, logs, rollout videos, and generated motion arrays should not be committed to git.

## Local Setup

The local environment is used for editing code, launching Modal jobs, logging experiments, and analyzing results. KimoLab generation and PPO training should run on Modal or another Linux NVIDIA GPU machine, not locally on macOS.

Create and activate the conda environment:

```bash
conda create -n text-to-torque python=3.11 -y
conda activate text-to-torque
```

Install local dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install modal wandb numpy pandas matplotlib scipy pyyaml tqdm rich ipython jupyter black ruff pytest
```

Verify the local environment:

```bash
python --version
python -m pip --version
python -c "import modal, wandb, numpy, pandas, matplotlib, yaml, tqdm; print('local env good')"
```

Set up Modal:

```bash
modal setup
modal --version
```

## Environment Variables and Secrets

Create a local `.env` file for secrets:

```bash
touch .env
```

Add your Hugging Face token:

```bash
HF_TOKEN=<your_huggingface_token>
```

Do not commit `.env` to git.

Load local environment variables:

```bash
set -a
source .env
set +a
```

Create the Hugging Face Modal secret:

```bash
modal secret create huggingface HF_TOKEN="$HF_TOKEN"
```

For W&B logging, log in locally:

```bash
wandb login
wandb --version
```

Create a W&B Modal secret if cloud training jobs need to log to W&B:

```bash
modal secret create wandb WANDB_API_KEY=<your_wandb_api_key>
```

## External Repo for Local Inspection

The external Unitree/mjlab repo is used as the underlying MuJoCo / Unitree G1 training pipeline. It is cloned locally only for reading code and debugging.

```bash
mkdir -p external
cd external
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
cd ..
```

The `external/` directory is ignored by git. Modal jobs clone and install external repositories inside the cloud image.

## KimoLab Modal Bring-Up

Run the default reference-motion generation job:

```bash
modal run modal_kimolab.py
```

Run generation only with a custom prompt:

```bash
modal run modal_kimolab.py \
  --prompt "A person walks forward" \
  --duration 4.0 \
  --seed 0 \
  --diffusion-steps 25 \
  --output-fps 50 \
  --render-reference \
  --no-train
```

Run a tiny PPO debug job:

```bash
modal run modal_kimolab.py \
  --prompt "A person walks forward" \
  --duration 4.0 \
  --seed 0 \
  --diffusion-steps 25 \
  --output-fps 50 \
  --render-reference \
  --train \
  --num-envs 128 \
  --max-iterations 20 \
  --save-interval 10 \
  --disable-terminations \
  --record-train-video
```

For longer jobs, use detached mode:

```bash
modal run --detach modal_kimolab.py --train
```

## Downloading Modal Outputs

Generated motions, videos, logs, and checkpoints are saved to the Modal Volume:

```text
text-to-torque-results
```

List files in the volume:

```bash
modal volume ls text-to-torque-results
```

List KimoLab outputs:

```bash
modal volume ls text-to-torque-results kimolab
```

Download KimoLab outputs locally:

```bash
mkdir -p motions/from_modal
modal volume get text-to-torque-results kimolab motions/from_modal
```

Download a specific run:

```bash
modal volume get text-to-torque-results kimolab/<run_id> motions/from_modal/<run_id>
```

Typical generated artifacts include:

```text
metadata.json
motion.csv
motion.npz
reference_motion.mp4
logs/
checkpoints/
```

## Quick Reference

```bash
# Generate motion only
modal run modal_kimolab.py --prompt "A person walks forward"

# Generate + train (detached)
modal run --detach modal_kimolab.py --prompt "A person walks forward" --train

# Generate + train + record video + wandb
modal run --detach modal_kimolab.py --prompt "A person walks forward" --train --record-train-video

# Key training flags
#   --max-iterations 20      PPO iterations (default 20)
#   --num-envs 128           parallel envs (default 128)
#   --save-interval 10       checkpoint every N iters
#   --disable-terminations   relax early termination (default True)
#   --duration 4.0           motion length in seconds
#   --seed 0                 random seed

# List outputs on Modal volume
modal volume ls text-to-torque-results kimolab

# List a specific run
modal volume ls text-to-torque-results kimolab/<run_id>

# Download a run locally
modal volume get text-to-torque-results kimolab/<run_id> ./motions/from_modal/kimolab/<run_id>

# Fresh terminal setup
conda activate text-to-torque && set -a && source .env && set +a
```