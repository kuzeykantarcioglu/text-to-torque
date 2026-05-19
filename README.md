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