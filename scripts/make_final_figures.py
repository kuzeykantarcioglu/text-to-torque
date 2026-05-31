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
}

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
                "step_offset": 1000,
            },
        ],
    )
    _plot_video_contact_sheet(args.figures_dir / "final_video_contact_sheet.png")

    print(f"Wrote {args.results_dir / 'final_experiment_summary.csv'}")
    print(f"Wrote {args.results_dir / 'final_experiment_summary.md'}")
    print(f"Wrote final figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
