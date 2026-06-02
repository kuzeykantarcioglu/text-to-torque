#!/usr/bin/env python3
"""Create final-report figures from downloaded KimoLab experiment artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROMPT_LABELS = {
    "A person walks forward": "Walk",
    "A person waves with their right hand": "Wave",
    "A person taps themselves on the head": "Tap head",
    "A person squats down and stands up": "Squat",
    "A person jumps": "Jump",
}

FINAL_OUTCOME_RUNS = [
    {
        "prompt": "A person walks forward",
        "condition": "loose_video",
        "display_condition": "Loose",
        "run_id": "1780201118_a-person-walks-forward_seed0_loose-video",
    },
    {
        "prompt": "A person walks forward",
        "condition": "strict",
        "display_condition": "Strict",
        "run_id": "1780200983_a-person-walks-forward_seed0_strict",
    },
    {
        "prompt": "A person walks forward",
        "condition": "curriculum_stage2_strict",
        "display_condition": "Curriculum",
        "run_id": "1780201423_a-person-walks-forward_seed0_curriculum",
    },
    {
        "prompt": "A person waves with their right hand",
        "condition": "loose",
        "display_condition": "Loose",
        "run_id": "1780189976_a-person-waves-with-their-right-hand_seed0",
    },
    {
        "prompt": "A person waves with their right hand",
        "condition": "strict",
        "display_condition": "Strict",
        "run_id": "1780201117_a-person-waves-with-their-right-hand_seed0_strict",
    },
    {
        "prompt": "A person taps themselves on the head",
        "condition": "loose",
        "display_condition": "Loose",
        "run_id": "1780189976_a-person-taps-themselves-on-the-head_seed0",
    },
    {
        "prompt": "A person taps themselves on the head",
        "condition": "strict",
        "display_condition": "Strict",
        "run_id": "1780201117_a-person-taps-themselves-on-the-head_seed0_strict",
    },
    {
        "prompt": "A person squats down and stands up",
        "condition": "loose_video",
        "display_condition": "Loose",
        "run_id": "1780201117_a-person-squats-down-and-stands-up_seed0_loose-video",
    },
    {
        "prompt": "A person squats down and stands up",
        "condition": "strict",
        "display_condition": "Strict",
        "run_id": "1780201117_a-person-squats-down-and-stands-up_seed0_strict",
    },
]

CONDITION_COLORS = {
    "Loose": "#4c78a8",
    "Strict": "#f58518",
    "Curriculum": "#54a24b",
    "Curriculum loose": "#72b7b2",
    "Curriculum strict": "#54a24b",
    "Loose seed 1": "#4c78a8",
    "Strict seed 1": "#f58518",
    "Jump loose": "#b279a2",
    "Threshold 1": "#e45756",
    "Threshold 2": "#b279a2",
    "Gradual thr=1": "#e45756",
    "Gradual thr=2": "#f58518",
    "Adaptive thr=0.5": "#e45756",
    "Adaptive thr=1": "#f58518",
    "Adaptive thr=2": "#54a24b",
    "Strict original": "#f58518",
    "Hold repair": "#4c78a8",
    "Smooth repair": "#72b7b2",
    "Hold + smooth repair": "#54a24b",
}

GRADUAL_CURRICULUM_RUN = "1780296901_a-person-squats-down-and-stands-up_seed0_squat-gradual-curriculum"
JUMP_LOOSE_RUN = "1780296901_a-person-jumps_seed0_jump-loose-extra"
ADAPTIVE_CALIBRATION_RUN = (
    "1780361838_a-person-squats-down-and-stands-up_seed0_squat-adaptive-calibrate"
)
REFERENCE_REPAIR_RUNS = [
    {
        "run_id": "1780201117_a-person-squats-down-and-stands-up_seed0_strict",
        "condition": "strict",
        "display_condition": "Strict original",
    },
    {
        "run_id": "1780362553_a-person-squats-down-and-stands-up_seed0_squat-repair-hold",
        "condition": "squat-repair-hold",
        "display_condition": "Hold repair",
    },
    {
        "run_id": "1780362553_a-person-squats-down-and-stands-up_seed0_squat-repair-smooth",
        "condition": "squat-repair-smooth",
        "display_condition": "Smooth repair",
    },
    {
        "run_id": "1780362553_a-person-squats-down-and-stands-up_seed0_squat-repair-hold-smooth",
        "condition": "squat-repair-hold-smooth",
        "display_condition": "Hold + smooth repair",
    },
]
CALIBRATED_THRESHOLD_RUNS = [
    {
        "run_id": "1780201117_a-person-squats-down-and-stands-up_seed0_loose-video",
        "condition": "loose_video",
        "display_condition": "Loose baseline",
    },
    {
        "run_id": "1780201117_a-person-squats-down-and-stands-up_seed0_strict",
        "condition": "strict",
        "display_condition": "Strict baseline",
    },
    {
        "run_id": GRADUAL_CURRICULUM_RUN,
        "condition": "curriculum_stage05_thr2",
        "display_condition": "Gradual curriculum, threshold 2",
    },
    {
        "run_id": GRADUAL_CURRICULUM_RUN,
        "condition": "curriculum_stage06_thr1",
        "display_condition": "Gradual curriculum, threshold 1",
    },
    {
        "run_id": "1780339437_a-person-squats-down-and-stands-up_seed0_squat-fixed-thr2",
        "condition": "squat-fixed-thr2",
        "display_condition": "Direct calibrated threshold 2",
    },
    {
        "run_id": "1780339437_a-person-squats-down-and-stands-up_seed0_squat-fixed-thr1",
        "condition": "squat-fixed-thr1",
        "display_condition": "Direct calibrated threshold 1",
    },
]
GRADUAL_STAGE_SPECS = [
    ("curriculum_stage01_loose", "thr=100", "#4c78a8"),
    ("curriculum_stage02_thr20", "thr=20", "#72b7b2"),
    ("curriculum_stage03_thr10", "thr=10", "#54a24b"),
    ("curriculum_stage04_thr5", "thr=5", "#eeca3b"),
    ("curriculum_stage05_thr2", "thr=2", "#f58518"),
    ("curriculum_stage06_thr1", "thr=1", "#e45756"),
    ("curriculum_stage07_strict", "strict", "#b279a2"),
]
ADAPTIVE_STAGE_SPECS = [
    ("adaptive_stage01_thr0p5", "thr=0.5", "#e45756"),
    ("adaptive_stage02_thr1", "thr=1", "#f58518"),
    ("adaptive_stage03_thr2", "thr=2", "#54a24b"),
]

VIDEO_SHEET_SPECS = [
    (
        "Walk loose",
        Path("motions/from_modal/1780201118_a-person-walks-forward_seed0_loose-video/logs/rsl_rl/g1_tracking/2026-05-31_04-21-56_1780201118_a-person-walks-forward_seed0_loose-video/videos/train/rl-video-step-40000.mp4"),
    ),
    (
        "Walk strict",
        Path("motions/from_modal/1780200983_a-person-walks-forward_seed0_strict/logs/rsl_rl/g1_tracking/2026-05-31_04-19-27_1780200983_a-person-walks-forward_seed0_strict/videos/train/rl-video-step-40000.mp4"),
    ),
    (
        "Walk curriculum",
        Path("motions/from_modal/1780201423_a-person-walks-forward_seed0_curriculum/logs/rsl_rl/g1_tracking/2026-05-31_04-46-33_1780201423_a-person-walks-forward_seed0_curriculum_stage2_strict/videos/train/rl-video-step-20000.mp4"),
    ),
    (
        "Squat loose",
        Path("motions/from_modal/1780201117_a-person-squats-down-and-stands-up_seed0_loose-video/logs/rsl_rl/g1_tracking/2026-05-31_04-21-52_1780201117_a-person-squats-down-and-stands-up_seed0_loose-video/videos/train/rl-video-step-40000.mp4"),
    ),
    (
        "Squat strict",
        Path("motions/from_modal/1780201117_a-person-squats-down-and-stands-up_seed0_strict/logs/rsl_rl/g1_tracking/2026-05-31_04-21-56_1780201117_a-person-squats-down-and-stands-up_seed0_strict/videos/train/rl-video-step-40000.mp4"),
    ),
    (
        "Wave strict",
        Path("motions/from_modal/1780201117_a-person-waves-with-their-right-hand_seed0_strict/logs/rsl_rl/g1_tracking/2026-05-31_04-23-46_1780201117_a-person-waves-with-their-right-hand_seed0_strict/videos/train/rl-video-step-40000.mp4"),
    ),
    (
        "Tap strict",
        Path("motions/from_modal/1780201117_a-person-taps-themselves-on-the-head_seed0_strict/logs/rsl_rl/g1_tracking/2026-05-31_04-23-45_1780201117_a-person-taps-themselves-on-the-head_seed0_strict/videos/train/rl-video-step-40000.mp4"),
    ),
]


def _clean_prompt(prompt: str) -> str:
    return PROMPT_LABELS.get(prompt, prompt.replace("A person ", ""))


def _smooth(values: pd.Series, window: int = 25) -> pd.Series:
    return values.rolling(window=window, min_periods=1, center=True).mean()


def _load_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    summary["prompt_label"] = summary["prompt"].map(_clean_prompt)
    return summary


def _load_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path, low_memory=False)
    metrics["prompt_label"] = metrics["prompt"].map(_clean_prompt)
    return metrics


def _load_diagnostics(path: Path) -> pd.DataFrame:
    diagnostics = pd.read_csv(path)
    diagnostics["prompt_label"] = diagnostics["prompt"].map(_clean_prompt)
    return diagnostics


def _select_final_outcomes(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in FINAL_OUTCOME_RUNS:
        mask = (
            (summary["run_id"] == spec["run_id"])
            & (summary["prompt"] == spec["prompt"])
            & (summary["condition"] == spec["condition"])
        )
        match = summary.loc[mask].copy()
        if match.empty:
            raise SystemExit(f"Missing final outcome row: {spec}")
        match["display_condition"] = spec["display_condition"]
        rows.append(match.iloc[0])

    final = pd.DataFrame(rows)
    final["prompt_label"] = final["prompt"].map(_clean_prompt)
    final["termination_total"] = (
        final.get("Episode_Termination/anchor_pos", 0).fillna(0)
        + final.get("Episode_Termination/ee_body_pos", 0).fillna(0)
    )
    return final


def _write_final_table(final: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    columns = [
        "prompt_label",
        "display_condition",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_joint_pos",
        "termination_total",
        "Perf/total_fps",
    ]
    table = final[columns].copy()
    table = table.rename(
        columns={
            "prompt_label": "motion",
            "display_condition": "condition",
            "Train/mean_reward": "final_reward",
            "Train/mean_episode_length": "final_episode_length",
            "Metrics/motion/error_body_pos": "body_pos_error",
            "Metrics/motion/error_joint_pos": "joint_pos_error",
            "Perf/total_fps": "steps_per_second",
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.6g")

    display = table.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(3)
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    md_path.write_text("\n".join(lines) + "\n")


def _write_extra_squat_table(summary: pd.DataFrame, csv_path: Path, md_path: Path) -> pd.DataFrame:
    specs = [
        {
            "run_id": "1780273575_a-person-squats-down-and-stands-up_seed0_squat-curriculum-extra",
            "condition": "curriculum_stage1_loose",
            "display_condition": "Seed 0 curriculum, loose stage",
        },
        {
            "run_id": "1780273575_a-person-squats-down-and-stands-up_seed0_squat-curriculum-extra",
            "condition": "curriculum_stage2_strict",
            "display_condition": "Seed 0 curriculum, strict stage",
        },
        {
            "run_id": "1780273575_a-person-squats-down-and-stands-up_seed1_squat-loose-seed1-extra",
            "condition": "squat-loose-seed1-extra",
            "display_condition": "Seed 1 loose",
        },
        {
            "run_id": "1780273575_a-person-squats-down-and-stands-up_seed1_squat-strict-seed1-extra",
            "condition": "squat-strict-seed1-extra",
            "display_condition": "Seed 1 strict",
        },
    ]

    rows = []
    for spec in specs:
        match = summary[
            (summary["run_id"] == spec["run_id"])
            & (summary["condition"] == spec["condition"])
        ].copy()
        if match.empty:
            print(f"[WARN] Missing extra squat row: {spec}")
            continue
        row = match.iloc[0].copy()
        row["display_condition"] = spec["display_condition"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    extra = pd.DataFrame(rows)
    extra["termination_total"] = (
        extra.get("Episode_Termination/anchor_pos", 0).fillna(0)
        + extra.get("Episode_Termination/ee_body_pos", 0).fillna(0)
    )

    columns = [
        "display_condition",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_joint_pos",
        "termination_total",
        "Perf/total_fps",
    ]
    table = extra[columns].rename(
        columns={
            "display_condition": "condition",
            "Train/mean_reward": "final_reward",
            "Train/mean_episode_length": "final_episode_length",
            "Metrics/motion/error_body_pos": "body_pos_error",
            "Metrics/motion/error_joint_pos": "joint_pos_error",
            "Perf/total_fps": "steps_per_second",
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.6g")

    display = table.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(3)
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    return extra


def _write_markdown_table(table: pd.DataFrame, output_path: Path) -> None:
    display = table.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(3)
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    output_path.write_text("\n".join(lines) + "\n")


def _write_gradual_curriculum_table(
    summary: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
) -> pd.DataFrame:
    rows = []
    for condition, label, _ in GRADUAL_STAGE_SPECS:
        match = summary[
            (summary["run_id"] == GRADUAL_CURRICULUM_RUN)
            & (summary["condition"] == condition)
        ].copy()
        if match.empty:
            print(f"[WARN] Missing gradual curriculum row: {condition}")
            continue
        row = match.iloc[0].copy()
        row["stage"] = label
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    gradual = pd.DataFrame(rows)
    gradual["termination_total"] = (
        gradual.get("Episode_Termination/anchor_pos", 0).fillna(0)
        + gradual.get("Episode_Termination/ee_body_pos", 0).fillna(0)
    )
    columns = [
        "stage",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_joint_pos",
        "termination_total",
        "Perf/total_fps",
    ]
    table = gradual[columns].rename(
        columns={
            "Train/mean_reward": "final_reward",
            "Train/mean_episode_length": "final_episode_length",
            "Metrics/motion/error_body_pos": "body_pos_error",
            "Metrics/motion/error_joint_pos": "joint_pos_error",
            "Perf/total_fps": "steps_per_second",
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.6g")
    _write_markdown_table(table, md_path)
    return gradual


def _write_adaptive_calibration_table(
    summary: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
) -> pd.DataFrame:
    rows = []
    for condition, label, _ in ADAPTIVE_STAGE_SPECS:
        match = summary[
            (summary["run_id"] == ADAPTIVE_CALIBRATION_RUN)
            & (summary["condition"] == condition)
        ].copy()
        if match.empty:
            print(f"[WARN] Missing adaptive calibration row: {condition}")
            continue
        row = match.iloc[0].copy()
        row["stage"] = label
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    adaptive = pd.DataFrame(rows)
    adaptive["termination_total"] = (
        adaptive.get("Episode_Termination/anchor_pos", 0).fillna(0)
        + adaptive.get("Episode_Termination/anchor_ori", 0).fillna(0)
        + adaptive.get("Episode_Termination/ee_body_pos", 0).fillna(0)
    )
    columns = [
        "stage",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_joint_pos",
        "Episode_Termination/anchor_pos",
        "Episode_Termination/anchor_ori",
        "Episode_Termination/ee_body_pos",
        "termination_total",
        "Perf/total_fps",
    ]
    table = adaptive[columns].rename(
        columns={
            "Train/mean_reward": "final_reward",
            "Train/mean_episode_length": "final_episode_length",
            "Metrics/motion/error_body_pos": "body_pos_error",
            "Metrics/motion/error_joint_pos": "joint_pos_error",
            "Episode_Termination/anchor_pos": "anchor_pos_terminations",
            "Episode_Termination/anchor_ori": "anchor_ori_terminations",
            "Episode_Termination/ee_body_pos": "ee_body_pos_terminations",
            "Perf/total_fps": "steps_per_second",
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.6g")
    _write_markdown_table(table, md_path)
    return adaptive


def _write_reference_repair_table(
    summary: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
) -> pd.DataFrame:
    rows = []
    for spec in REFERENCE_REPAIR_RUNS:
        match = summary[
            (summary["run_id"] == spec["run_id"])
            & (summary["condition"] == spec["condition"])
        ].copy()
        if match.empty:
            print(f"[WARN] Missing reference-repair row: {spec}")
            continue
        row = match.iloc[0].copy()
        row["display_condition"] = spec["display_condition"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    repair = pd.DataFrame(rows)
    repair["termination_total"] = (
        repair.get("Episode_Termination/anchor_pos", 0).fillna(0)
        + repair.get("Episode_Termination/anchor_ori", 0).fillna(0)
        + repair.get("Episode_Termination/ee_body_pos", 0).fillna(0)
    )
    columns = [
        "display_condition",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_joint_pos",
        "Episode_Termination/anchor_pos",
        "Episode_Termination/anchor_ori",
        "Episode_Termination/ee_body_pos",
        "termination_total",
        "Perf/total_fps",
    ]
    table = repair[columns].rename(
        columns={
            "display_condition": "condition",
            "Train/mean_reward": "final_reward",
            "Train/mean_episode_length": "final_episode_length",
            "Metrics/motion/error_body_pos": "body_pos_error",
            "Metrics/motion/error_joint_pos": "joint_pos_error",
            "Episode_Termination/anchor_pos": "anchor_pos_terminations",
            "Episode_Termination/anchor_ori": "anchor_ori_terminations",
            "Episode_Termination/ee_body_pos": "ee_body_pos_terminations",
            "Perf/total_fps": "steps_per_second",
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.6g")
    _write_markdown_table(table, md_path)
    return repair


def _write_jump_table(summary: pd.DataFrame, csv_path: Path, md_path: Path) -> pd.DataFrame:
    match = summary[
        (summary["run_id"] == JUMP_LOOSE_RUN)
        & (summary["condition"] == "jump-loose-extra")
    ].copy()
    if match.empty:
        print("[WARN] Missing jump training row")
        return pd.DataFrame()

    jump = match.copy()
    jump["termination_total"] = (
        jump.get("Episode_Termination/anchor_pos", 0).fillna(0)
        + jump.get("Episode_Termination/ee_body_pos", 0).fillna(0)
    )
    table = jump[
        [
            "Train/mean_reward",
            "Train/mean_episode_length",
            "Metrics/motion/error_body_pos",
            "Metrics/motion/error_joint_pos",
            "termination_total",
            "Perf/total_fps",
        ]
    ].rename(
        columns={
            "Train/mean_reward": "final_reward",
            "Train/mean_episode_length": "final_episode_length",
            "Metrics/motion/error_body_pos": "body_pos_error",
            "Metrics/motion/error_joint_pos": "joint_pos_error",
            "Perf/total_fps": "steps_per_second",
        }
    )
    table.insert(0, "motion", "Jump")
    table.insert(1, "condition", "Loose")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.6g")
    _write_markdown_table(table, md_path)
    return jump


def _write_calibrated_threshold_table(
    summary: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
) -> pd.DataFrame:
    rows = []
    for spec in CALIBRATED_THRESHOLD_RUNS:
        match = summary[
            (summary["run_id"] == spec["run_id"])
            & (summary["condition"] == spec["condition"])
        ].copy()
        if match.empty:
            print(f"[WARN] Missing calibrated-threshold row: {spec}")
            continue
        row = match.iloc[0].copy()
        row["display_condition"] = spec["display_condition"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    calibrated = pd.DataFrame(rows)
    calibrated["termination_total"] = (
        calibrated.get("Episode_Termination/anchor_pos", 0).fillna(0)
        + calibrated.get("Episode_Termination/anchor_ori", 0).fillna(0)
        + calibrated.get("Episode_Termination/ee_body_pos", 0).fillna(0)
    )
    columns = [
        "display_condition",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_joint_pos",
        "Episode_Termination/anchor_pos",
        "Episode_Termination/anchor_ori",
        "Episode_Termination/ee_body_pos",
        "termination_total",
        "Perf/total_fps",
    ]
    table = calibrated[columns].rename(
        columns={
            "display_condition": "condition",
            "Train/mean_reward": "final_reward",
            "Train/mean_episode_length": "final_episode_length",
            "Metrics/motion/error_body_pos": "body_pos_error",
            "Metrics/motion/error_joint_pos": "joint_pos_error",
            "Episode_Termination/anchor_pos": "anchor_pos_terminations",
            "Episode_Termination/anchor_ori": "anchor_ori_terminations",
            "Episode_Termination/ee_body_pos": "ee_body_pos_terminations",
            "Perf/total_fps": "steps_per_second",
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.6g")
    _write_markdown_table(table, md_path)
    return calibrated


def _plot_outcome_bars(final: pd.DataFrame, output_path: Path) -> None:
    prompt_order = ["Walk", "Wave", "Tap head", "Squat"]
    condition_order = ["Loose", "Strict", "Curriculum"]
    x = np.arange(len(prompt_order))
    width = 0.23

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    metrics = [
        ("Train/mean_reward", "Final reward", None),
        ("Train/mean_episode_length", "Episode length", (0, 260)),
        ("termination_total", "Termination count", None),
    ]

    for axis, (column, title, ylim) in zip(axes, metrics, strict=True):
        for offset_index, condition in enumerate(condition_order):
            values = []
            for prompt in prompt_order:
                row = final[
                    (final["prompt_label"] == prompt)
                    & (final["display_condition"] == condition)
                ]
                values.append(float(row[column].iloc[0]) if not row.empty else np.nan)
            offset = (offset_index - 1) * width
            axis.bar(
                x + offset,
                values,
                width=width,
                label=condition,
                color=CONDITION_COLORS[condition],
            )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(prompt_order, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if ylim:
            axis.set_ylim(*ylim)

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Final PPO tracking outcomes by termination setting")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _curve(
    metrics: pd.DataFrame,
    run_id: str,
    condition: str,
    tag: str,
    step_offset: int = 0,
) -> pd.DataFrame:
    subset = metrics[
        (metrics["run_id"] == run_id)
        & (metrics["condition"] == condition)
        & (metrics["tag"] == tag)
    ].copy()
    subset = subset.sort_values("step")
    subset["plot_step"] = subset["step"] + step_offset
    subset["smooth_value"] = _smooth(subset["value"])
    return subset


def _plot_comparison_curves(
    metrics: pd.DataFrame,
    output_path: Path,
    title: str,
    specs: list[dict[str, object]],
) -> None:
    plot_tags = [
        ("Train/mean_reward", "Reward"),
        ("Train/mean_episode_length", "Episode length"),
        ("Metrics/motion/error_body_pos", "Body position error"),
    ]
    fig, axes = plt.subplots(len(plot_tags), 1, figsize=(8.5, 7), sharex=True, constrained_layout=True)

    for axis, (tag, tag_title) in zip(axes, plot_tags, strict=True):
        for spec in specs:
            curve = _curve(
                metrics,
                run_id=str(spec["run_id"]),
                condition=str(spec["condition"]),
                tag=tag,
                step_offset=int(spec.get("step_offset", 0)),
            )
            if curve.empty:
                continue
            label = str(spec["label"])
            axis.plot(
                curve["plot_step"],
                curve["smooth_value"],
                linewidth=1.8,
                label=label,
                color=CONDITION_COLORS.get(label, spec.get("color")),
            )
        axis.set_title(tag_title)
        axis.grid(alpha=0.25)
        axis.set_ylabel("Value")

    axes[-1].set_xlabel("Training iteration")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_diagnostics(diagnostics: pd.DataFrame, output_path: Path) -> None:
    final_runs = [
        "1780201118_a-person-walks-forward_seed0_loose-video",
        "1780201117_a-person-waves-with-their-right-hand_seed0_strict",
        "1780201117_a-person-taps-themselves-on-the-head_seed0_strict",
        "1780201117_a-person-squats-down-and-stands-up_seed0_loose-video",
    ]
    rows = diagnostics[diagnostics["run_id"].isin(final_runs)].copy()
    rows["prompt_label"] = rows["prompt"].map(_clean_prompt)
    rows = rows.set_index("prompt_label").loc[["Walk", "Wave", "Tap head", "Squat"]].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    metrics = [
        ("root_z_range", "Root height range", "m"),
        ("root_speed_max", "Max root speed", "m/s"),
        ("joint_accel_abs_max", "Max joint acceleration", "rad/s^2"),
    ]
    for axis, (column, title, ylabel) in zip(axes, metrics, strict=True):
        axis.bar(rows["prompt_label"], rows[column], color="#4c78a8")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Reference-motion diagnostics before PPO")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_video_contact_sheet(output_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        print("[WARN] Skipping video contact sheet: ffmpeg is not on PATH")
        return

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[WARN] Skipping video contact sheet: Pillow is not installed")
        return

    missing = [str(path) for _, path in VIDEO_SHEET_SPECS if not path.exists()]
    if missing:
        print("[WARN] Skipping video contact sheet because videos are missing:")
        for path in missing:
            print(f"  {path}")
        return

    thumb_w, thumb_h = 320, 200
    label_h = 34
    padding = 14
    columns = 4
    rows = 2
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for index, (label, video_path) in enumerate(VIDEO_SHEET_SPECS):
            frame_path = tmpdir / f"frame_{index}.jpg"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    "1.2",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(frame_path),
                ],
                check=True,
            )
            image = Image.open(frame_path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
            canvas.paste(image, ((thumb_w - image.width) // 2, 0))

            draw = ImageDraw.Draw(canvas)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_x = (thumb_w - (bbox[2] - bbox[0])) // 2
            draw.text((text_x, thumb_h + 7), label, fill="black", font=font)
            frames.append(canvas)

    blank = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
    while len(frames) < rows * columns:
        frames.append(blank.copy())

    sheet = Image.new(
        "RGB",
        (
            columns * thumb_w + (columns + 1) * padding,
            rows * (thumb_h + label_h) + (rows + 1) * padding,
        ),
        "white",
    )
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        x = padding + column * (thumb_w + padding)
        y = padding + row * (thumb_h + label_h + padding)
        sheet.paste(frame, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=Path("results/training_summary.csv"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("results/training_metrics.csv"))
    parser.add_argument("--diagnostics-csv", type=Path, default=Path("results/motion_diagnostics.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _load_summary(args.summary_csv)
    metrics = _load_metrics(args.metrics_csv)
    diagnostics = _load_diagnostics(args.diagnostics_csv)

    final = _select_final_outcomes(summary)
    _write_final_table(
        final,
        args.results_dir / "final_experiment_summary.csv",
        args.results_dir / "final_experiment_summary.md",
    )
    _plot_outcome_bars(final, args.figures_dir / "final_outcome_summary.png")
    _plot_diagnostics(diagnostics, args.figures_dir / "final_reference_diagnostics.png")
    _plot_comparison_curves(
        metrics,
        args.figures_dir / "final_squat_termination_curves.png",
        "Squat tracking: strict termination prevents full-horizon learning",
        [
            {
                "run_id": "1780201117_a-person-squats-down-and-stands-up_seed0_loose-video",
                "condition": "loose_video",
                "label": "Loose",
            },
            {
                "run_id": "1780201117_a-person-squats-down-and-stands-up_seed0_strict",
                "condition": "strict",
                "label": "Strict",
            },
        ],
    )
    _plot_comparison_curves(
        metrics,
        args.figures_dir / "final_walk_curriculum_curves.png",
        "Walk tracking: loose, strict, and two-stage curriculum",
        [
            {
                "run_id": "1780201118_a-person-walks-forward_seed0_loose-video",
                "condition": "loose_video",
                "label": "Loose",
            },
            {
                "run_id": "1780200983_a-person-walks-forward_seed0_strict",
                "condition": "strict",
                "label": "Strict",
            },
            {
                "run_id": "1780201423_a-person-walks-forward_seed0_curriculum",
                "condition": "curriculum_stage1_loose",
                "label": "Curriculum loose",
            },
            {
                "run_id": "1780201423_a-person-walks-forward_seed0_curriculum",
                "condition": "curriculum_stage2_strict",
                "label": "Curriculum strict",
            },
        ],
    )
    extra_squat = _write_extra_squat_table(
        summary,
        args.results_dir / "extra_squat_summary.csv",
        args.results_dir / "extra_squat_summary.md",
    )
    if not extra_squat.empty:
        _plot_comparison_curves(
            metrics,
            args.figures_dir / "final_squat_extra_curves.png",
            "Extra squat runs: seed sensitivity and curriculum collapse",
            [
                {
                    "run_id": "1780273575_a-person-squats-down-and-stands-up_seed0_squat-curriculum-extra",
                    "condition": "curriculum_stage1_loose",
                    "label": "Curriculum loose",
                },
                {
                    "run_id": "1780273575_a-person-squats-down-and-stands-up_seed0_squat-curriculum-extra",
                    "condition": "curriculum_stage2_strict",
                    "label": "Curriculum strict",
                },
                {
                    "run_id": "1780273575_a-person-squats-down-and-stands-up_seed1_squat-loose-seed1-extra",
                    "condition": "squat-loose-seed1-extra",
                    "label": "Loose seed 1",
                },
                {
                    "run_id": "1780273575_a-person-squats-down-and-stands-up_seed1_squat-strict-seed1-extra",
                    "condition": "squat-strict-seed1-extra",
                    "label": "Strict seed 1",
                },
            ],
        )
    gradual = _write_gradual_curriculum_table(
        summary,
        args.results_dir / "gradual_curriculum_summary.csv",
        args.results_dir / "gradual_curriculum_summary.md",
    )
    if not gradual.empty:
        _plot_comparison_curves(
            metrics,
            args.figures_dir / "final_gradual_curriculum_curves.png",
            "Squat seed 0: gradual termination curriculum still collapses at strict transfer",
            [
                {
                    "run_id": GRADUAL_CURRICULUM_RUN,
                    "condition": condition,
                    "label": label,
                    "color": color,
                }
                for condition, label, color in GRADUAL_STAGE_SPECS
            ],
        )

    calibrated = _write_calibrated_threshold_table(
        summary,
        args.results_dir / "calibrated_threshold_summary.csv",
        args.results_dir / "calibrated_threshold_summary.md",
    )
    if not calibrated.empty:
        _plot_comparison_curves(
            metrics,
            args.figures_dir / "final_calibrated_threshold_curves.png",
            "Squat seed 0: calibrated termination thresholds",
            [
                {
                    "run_id": "1780201117_a-person-squats-down-and-stands-up_seed0_loose-video",
                    "condition": "loose_video",
                    "label": "Loose",
                },
                {
                    "run_id": "1780201117_a-person-squats-down-and-stands-up_seed0_strict",
                    "condition": "strict",
                    "label": "Strict",
                },
                {
                    "run_id": GRADUAL_CURRICULUM_RUN,
                    "condition": "curriculum_stage05_thr2",
                    "label": "Gradual thr=2",
                },
                {
                    "run_id": GRADUAL_CURRICULUM_RUN,
                    "condition": "curriculum_stage06_thr1",
                    "label": "Gradual thr=1",
                },
                {
                    "run_id": "1780339437_a-person-squats-down-and-stands-up_seed0_squat-fixed-thr2",
                    "condition": "squat-fixed-thr2",
                    "label": "Threshold 2",
                },
                {
                    "run_id": "1780339437_a-person-squats-down-and-stands-up_seed0_squat-fixed-thr1",
                    "condition": "squat-fixed-thr1",
                    "label": "Threshold 1",
                },
            ],
        )

    adaptive = _write_adaptive_calibration_table(
        summary,
        args.results_dir / "adaptive_calibration_summary.csv",
        args.results_dir / "adaptive_calibration_summary.md",
    )
    if not adaptive.empty:
        _plot_comparison_curves(
            metrics,
            args.figures_dir / "final_adaptive_calibration_curves.png",
            "Squat seed 0: adaptive termination calibration",
            [
                {
                    "run_id": ADAPTIVE_CALIBRATION_RUN,
                    "condition": condition,
                    "label": f"Adaptive {label}",
                    "color": color,
                }
                for condition, label, color in ADAPTIVE_STAGE_SPECS
            ],
        )

    repair = _write_reference_repair_table(
        summary,
        args.results_dir / "reference_repair_summary.csv",
        args.results_dir / "reference_repair_summary.md",
    )
    if not repair.empty:
        _plot_comparison_curves(
            metrics,
            args.figures_dir / "final_reference_repair_curves.png",
            "Squat seed 0: strict training with repaired references",
            [
                {
                    "run_id": spec["run_id"],
                    "condition": spec["condition"],
                    "label": spec["display_condition"],
                }
                for spec in REFERENCE_REPAIR_RUNS
            ],
        )

    jump = _write_jump_table(
        summary,
        args.results_dir / "jump_training_summary.csv",
        args.results_dir / "jump_training_summary.md",
    )
    if not jump.empty:
        _plot_comparison_curves(
            metrics,
            args.figures_dir / "final_jump_training_curves.png",
            "Jump tracking under loose termination",
            [
                {
                    "run_id": JUMP_LOOSE_RUN,
                    "condition": "jump-loose-extra",
                    "label": "Jump loose",
                }
            ],
        )
    _plot_video_contact_sheet(args.figures_dir / "final_video_contact_sheet.png")

    print(f"Wrote {args.results_dir / 'final_experiment_summary.csv'}")
    print(f"Wrote {args.results_dir / 'final_experiment_summary.md'}")
    print(f"Wrote final figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
