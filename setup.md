# Setup

This file describes the local development setup for TextToTorque.

The local environment is used for editing code, launching Modal jobs, logging experiments with Weights & Biases, and running analysis scripts. Full MuJoCo / Unitree G1 training runs on Modal or another Linux GPU machine, not locally on macOS.

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

## 3. Verify the Local Environment

```bash
python --version
python -m pip --version
python -c "import modal, wandb, numpy, pandas, matplotlib, yaml, tqdm; print('local env good')"
```

## 4. Set Up Modal

```bash
modal setup
modal --version
```

Modal is used to launch GPU jobs in the cloud.

## 5. Set Up Weights & Biases

Log in locally:

```bash
wandb login
wandb --version
```

For Modal jobs, create a W&B secret so cloud training runs can log metrics:

```bash
modal secret create wandb WANDB_API_KEY=<your_wandb_api_key>
```

Do not commit your W&B API key to git.

In Modal scripts, attach the secret to training functions with:

```python
secrets=[modal.Secret.from_name("wandb")]
```

For smoke tests where logging is not needed, W&B can be disabled with:

```python
.env({"WANDB_MODE": "disabled"})
```

## 6. External Repo for Local Inspection

The external Unitree/mjlab repo is used as the underlying MuJoCo / Unitree G1 training pipeline. It is cloned locally only for reading code and debugging.

```bash
mkdir -p external
cd external
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
cd ..
```

## 7. Project Structure

```text
external/      External repos for local inspection only
scripts/       Project scripts and diagnostics
configs/       Experiment configs
experiments/   Per-run notes and metadata
results/       Figures, tables, and small result files
motions/       Generated or processed motion files
```

The `external/` directory is ignored by git. Modal jobs clone/install external repositories inside the cloud image.

Large files such as checkpoints, logs, rollout videos, and generated motion arrays should not be committed to git.

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

Run a Modal script:

```bash
modal run <script_name>.py
```

Run a Modal script detached:

```bash
modal run --detach <script_name>.py
```

Commit setup changes:

```bash
git add setup.md
git commit -m "Add setup instructions"
git push
```