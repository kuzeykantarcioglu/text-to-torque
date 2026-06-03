#!/usr/bin/env python3
"""Rank generated references by cheap pre-training feasibility diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "root_z_range",
    "root_speed_max",
    "root_accel_max",
    "joint_vel_abs_max",
    "joint_accel_abs_max",
    "body_ang_speed_max",
]


def _robust_normalize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    scale = q3 - q1
    if not np.isfinite(scale) or scale < 1e-9:
        scale = numeric.std()
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return (numeric - median) / scale


def _stable_random_choice(indices: Iterable[int], key: str) -> int:
    ordered = list(indices)
    if not ordered:
        raise ValueError("Cannot choose from an empty index set")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def _read_prompt_suite(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _motion_id(prompt: str) -> str:
    text = prompt.lower()
    if "turn" in text and "walk" in text:
        return "turn_walk"
    if "walks forward" in text:
        return "walk_forward"
    if "waves" in text and "right hand" in text:
        return "wave_right"
    if "tap" in text and "head" in text:
        return "tap_head"
    if "squat" in text:
        return "squat_stand"
    if "jump" in text:
        return "jump"
    if "roll" in text:
        return "roll"
    if "backflip" in text:
        return "backflip"
    if "cartwheel" in text:
        return "cartwheel"
    return "-".join(text.replace("a person", "").split())[:48]


def rank_references(
    diagnostics: pd.DataFrame,
    prompt_suite: pd.DataFrame,
    metrics: list[str],
    run_label_filter: str,
) -> pd.DataFrame:
    data = diagnostics.copy()
    if run_label_filter:
        labels = data["run_label"].fillna("").astype(str)
        data = data[labels.str.contains(run_label_filter, regex=False)].copy()

    if prompt_suite.empty:
        prompt_keys = data[["prompt"]].drop_duplicates().copy()
        prompt_keys["difficulty"] = ""
    else:
        prompt_keys = prompt_suite[["prompt", "difficulty"]].drop_duplicates()

    data = data.merge(prompt_keys, on="prompt", how="left", suffixes=("", "_suite"))
    if "difficulty_suite" in data:
        data["difficulty"] = data["difficulty"].fillna(data["difficulty_suite"])
        data = data.drop(columns=["difficulty_suite"])
    data["difficulty"] = data["difficulty"].fillna("")
    data["motion_id"] = data["prompt"].map(_motion_id)

    present_metrics = [metric for metric in metrics if metric in data.columns]
    if not present_metrics:
        raise SystemExit("None of the requested diagnostic metrics were present")

    for metric in present_metrics:
        data[f"{metric}_score"] = _robust_normalize(data[metric]).clip(lower=-3, upper=6)

    score_columns = [f"{metric}_score" for metric in present_metrics]
    data["feasibility_score"] = data[score_columns].sum(axis=1)
    data["feasibility_score"] = data["feasibility_score"].round(6)

    rows = []
    for prompt, group in data.groupby("prompt", sort=True):
        ranked = group.sort_values(
            ["feasibility_score", "joint_accel_abs_max", "root_accel_max", "run_id"],
            na_position="last",
        ).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        random_index = _stable_random_choice(ranked.index, prompt)
        best_index = ranked.index[0]
        worst_index = ranked.index[-1]
        selections = []
        for index in ranked.index:
            if index == best_index:
                selections.append("best")
            elif index == worst_index:
                selections.append("worst")
            elif index == random_index:
                selections.append("random")
            else:
                selections.append("")
        ranked["selection"] = selections
        rows.append(ranked)

    output = pd.concat(rows, ignore_index=True)
    ordered_columns = [
        "motion_id",
        "difficulty",
        "prompt",
        "seed",
        "selection",
        "rank",
        "feasibility_score",
        "run_id",
        "run_label",
        "root_z_range",
        "root_speed_max",
        "root_accel_max",
        "joint_vel_abs_max",
        "joint_accel_abs_max",
        "body_ang_speed_max",
        "duration_actual_s",
        "has_reference_video",
    ]
    present = [column for column in ordered_columns if column in output.columns]
    remaining = [column for column in output.columns if column not in present]
    return output[present + remaining]


def write_markdown(rows: pd.DataFrame, output_path: Path) -> None:
    selected = rows[rows["selection"].astype(str) != ""].copy()
    if selected.empty:
        selected = rows.copy()
    display_columns = [
        "motion_id",
        "selection",
        "seed",
        "rank",
        "feasibility_score",
        "root_z_range",
        "root_accel_max",
        "joint_accel_abs_max",
        "run_id",
    ]
    display_columns = [column for column in display_columns if column in selected.columns]
    display = selected[display_columns].copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(3)

    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def write_selected_launch_csv(rows: pd.DataFrame, output_path: Path) -> None:
    selected = rows[rows["selection"].astype(str) != ""].copy()
    columns = ["motion_id", "difficulty", "prompt", "duration_requested_s", "seed", "selection", "run_id"]
    present = [column for column in columns if column in selected.columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected[present].to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-csv", type=Path, default=Path("results/motion_diagnostics.csv"))
    parser.add_argument("--prompt-suite", type=Path, default=Path("prompts/seed_sweep_suite.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/reference_rankings.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("results/reference_rankings.md"))
    parser.add_argument("--selected-csv", type=Path, default=Path("results/selected_references.csv"))
    parser.add_argument(
        "--run-label-filter",
        default="",
        help="Only rank diagnostics whose run_label contains this string.",
    )
    parser.add_argument("--metric", action="append", default=[], help="Diagnostic metric to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = pd.read_csv(args.diagnostics_csv)
    prompt_suite = _read_prompt_suite(args.prompt_suite)
    metrics = args.metric or DEFAULT_METRICS
    rankings = rank_references(diagnostics, prompt_suite, metrics, args.run_label_filter)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output_csv, index=False, float_format="%.6g")
    write_markdown(rankings, args.output_md)
    write_selected_launch_csv(rankings, args.selected_csv)

    print(f"Wrote rankings to {args.output_csv}")
    print(f"Wrote selected references to {args.selected_csv}")


if __name__ == "__main__":
    main()
