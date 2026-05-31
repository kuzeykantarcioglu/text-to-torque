#!/usr/bin/env python3
"""Export KimoLab TensorBoard training metrics to CSV and report plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_joint_pos",
    "Metrics/motion/error_anchor_pos",
    "Episode_Termination/anchor_pos",
    "Episode_Termination/anchor_ori",
    "Episode_Termination/ee_body_pos",
    "Perf/total_fps",
]


def _run_metadata(run_dir: Path) -> dict[str, str]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open() as f:
        return json.load(f)


def _event_log_name(event_path: Path, run_dir: Path) -> str:
    try:
        return event_path.relative_to(run_dir).parent.name
    except ValueError:
        return event_path.parent.name


def _effective_condition(event_log_name: str, metadata: dict[str, object]) -> tuple[str, bool | None]:
    if "stage1_loose" in event_log_name:
        return "curriculum_stage1_loose", True
    if "stage2_strict" in event_log_name:
        return "curriculum_stage2_strict", False

    disable_terminations = metadata.get("disable_terminations", None)
    run_label = str(metadata.get("run_label", "") or "")
    if run_label:
        return run_label, disable_terminations if isinstance(disable_terminations, bool) else None
    if isinstance(disable_terminations, bool):
        return ("loose" if disable_terminations else "strict"), disable_terminations
    return "", None


def _event_rows(run_dir: Path, tags: list[str]) -> list[dict[str, object]]:
    metadata = _run_metadata(run_dir)
    rows: list[dict[str, object]] = []

    for event_path in sorted(run_dir.glob("logs/**/events.out.tfevents.*")):
        event_log = _event_log_name(event_path, run_dir)
        condition, effective_disable_terminations = _effective_condition(event_log, metadata)
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        available_tags = set(accumulator.Tags().get("scalars", []))

        for tag in tags:
            if tag not in available_tags:
                continue
            for event in accumulator.Scalars(tag):
                rows.append(
                    {
                        "run_id": run_dir.name,
                        "event_log": event_log,
                        "run_label": metadata.get("run_label", ""),
                        "condition": condition,
                        "disable_terminations": effective_disable_terminations,
                        "prompt": metadata.get("prompt", run_dir.name),
                        "tag": tag,
                        "step": event.step,
                        "wall_time": event.wall_time,
                        "value": event.value,
                    }
                )

    return rows


def _friendly_prompt(prompt: str) -> str:
    prompt = prompt.removeprefix("A person ").removeprefix("a person ")
    return prompt[:1].upper() + prompt[1:]


def _write_summary(metrics: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    latest = (
        metrics.sort_values(["run_id", "tag", "step"])
        .groupby(
            [
                "run_id",
                "event_log",
                "run_label",
                "condition",
                "disable_terminations",
                "prompt",
                "tag",
            ],
            as_index=False,
            dropna=False,
        )
        .tail(1)
    )
    summary = latest.pivot(
        index=[
            "run_id",
            "event_log",
            "run_label",
            "condition",
            "disable_terminations",
            "prompt",
        ],
        columns="tag",
        values="value",
    ).reset_index()
    summary.columns.name = None

    ordered = [
        "run_id",
        "event_log",
        "condition",
        "run_label",
        "disable_terminations",
        "prompt",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_joint_pos",
        "Metrics/motion/error_anchor_pos",
        "Episode_Termination/anchor_pos",
        "Perf/total_fps",
    ]
    present = [column for column in ordered if column in summary.columns]
    remaining = [column for column in summary.columns if column not in present]
    summary = summary[present + remaining]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def _write_summary_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    display = summary.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(3)

    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n")


def _plot_curves(metrics: pd.DataFrame, output_path: Path) -> None:
    plot_tags = [
        ("Train/mean_reward", "Mean Reward"),
        ("Metrics/motion/error_body_pos", "Body Position Error"),
        ("Metrics/motion/error_joint_pos", "Joint Position Error"),
    ]

    fig, axes = plt.subplots(len(plot_tags), 1, figsize=(10, 8), sharex=True)
    for axis, (tag, title) in zip(axes, plot_tags, strict=True):
        subset = metrics[metrics["tag"] == tag]
        for (run_id, condition, prompt, event_log), group in subset.groupby(
            ["run_id", "condition", "prompt", "event_log"], dropna=False
        ):
            group = group.sort_values("step")
            label = _friendly_prompt(prompt)
            if condition:
                label = f"{label} [{condition}]"
            elif subset["prompt"].value_counts().get(prompt, 0) > len(group):
                label = f"{label} [{run_id[:10]}]"
            if "stage" in event_log and condition not in label:
                label = f"{label} [{event_log.split('_')[-2]} {event_log.split('_')[-1]}]"
            axis.plot(group["step"], group["value"], linewidth=1.4, label=label)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_ylabel("Value")

    axes[-1].set_xlabel("Training Iteration")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motions-dir", type=Path, default=Path("motions/from_modal"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("results/training_metrics.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/training_summary.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("results/training_summary.md"))
    parser.add_argument("--figure", type=Path, default=Path("figures/training_curves.png"))
    parser.add_argument(
        "--run-prefix",
        action="append",
        default=[],
        help="Only include run IDs with this prefix. May be passed multiple times.",
    )
    parser.add_argument("--tag", action="append", default=[], help="Extra TensorBoard scalar tag to export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tags = DEFAULT_TAGS + [tag for tag in args.tag if tag not in DEFAULT_TAGS]

    rows: list[dict[str, object]] = []
    for run_dir in sorted(args.motions_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if args.run_prefix and not any(run_dir.name.startswith(prefix) for prefix in args.run_prefix):
            continue
        rows.extend(_event_rows(run_dir, tags))

    if not rows:
        raise SystemExit(f"No TensorBoard metrics found under {args.motions_dir}")

    metrics = pd.DataFrame(rows)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_csv, index=False, float_format="%.6g")

    summary = _write_summary(metrics, args.summary_csv)
    _write_summary_markdown(summary, args.summary_md)
    _plot_curves(metrics, args.figure)

    print(f"Wrote {len(metrics)} metric rows to {args.metrics_csv}")
    print(f"Wrote summary to {args.summary_csv} and {args.summary_md}")
    print(f"Wrote figure to {args.figure}")


if __name__ == "__main__":
    main()
