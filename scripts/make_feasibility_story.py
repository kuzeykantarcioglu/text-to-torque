#!/usr/bin/env python3
"""Build figures for the feasibility-aware reference selection story."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MOTION_LABELS = {
    "walk_forward": "Walk",
    "tap_head": "Tap head",
    "squat_stand": "Squat",
    "turn_walk": "Turn + walk",
    "jump": "Jump",
    "roll": "Roll",
    "wave_right": "Wave",
    "backflip": "Backflip",
    "cartwheel": "Cartwheel",
}


def _motion_label(value: str) -> str:
    return MOTION_LABELS.get(value, value.replace("_", " ").title())


def _write_markdown(table: pd.DataFrame, output_path: Path) -> None:
    display = table.copy()
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


def _plot_reference_rankings(rankings: pd.DataFrame, output_path: Path) -> None:
    rankings = rankings.copy()
    if rankings.empty:
        return
    rankings["motion_label"] = rankings["motion_id"].map(_motion_label)
    motion_order = [
        label for label in ["Walk", "Tap head", "Squat", "Turn + walk", "Jump", "Roll"]
        if label in set(rankings["motion_label"])
    ]

    fig, axes = plt.subplots(
        len(motion_order),
        1,
        figsize=(9.2, max(3.2, 1.85 * len(motion_order))),
        sharex=False,
    )
    if len(motion_order) == 1:
        axes = [axes]

    colors = {"best": "#54a24b", "random": "#4c78a8", "worst": "#e45756", "": "#bab0ab"}
    for axis, motion_label in zip(axes, motion_order, strict=True):
        group = rankings[rankings["motion_label"] == motion_label].sort_values("rank")
        labels = [f"s{int(seed)}" if pd.notna(seed) else str(run_id)[:6] for seed, run_id in zip(group["seed"], group["run_id"], strict=True)]
        bar_colors = [colors.get(str(selection), "#bab0ab") for selection in group["selection"].fillna("")]
        axis.bar(labels, group["feasibility_score"], color=bar_colors)
        axis.set_ylabel(motion_label)
        axis.grid(axis="y", alpha=0.25)
        for x, (_, row) in enumerate(group.iterrows()):
            selection = str(row.get("selection", ""))
            if selection:
                axis.text(
                    x,
                    float(row["feasibility_score"]),
                    selection,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    axes[-1].set_xlabel("Generated seed")
    fig.suptitle("Best-of-N reference selection from pre-training diagnostics")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.91, bottom=0.08, hspace=0.65)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _recommendations(rankings: pd.DataFrame) -> pd.DataFrame:
    selected = rankings[rankings["selection"].isin(["best", "worst"])].copy()
    if selected.empty:
        return pd.DataFrame()
    rows = []
    for motion_id, group in rankings.groupby("motion_id", sort=True):
        best = group.sort_values("feasibility_score").iloc[0]
        worst = group.sort_values("feasibility_score").iloc[-1]
        spread = float(worst["feasibility_score"] - best["feasibility_score"])
        score = float(best["feasibility_score"])
        if score <= -1.0:
            action = "strict PPO"
        elif score <= 2.0:
            action = "adaptive termination"
        else:
            action = "temporal repair or reject seed"
        rows.append(
            {
                "motion": _motion_label(motion_id),
                "best_seed": best.get("seed", ""),
                "best_score": score,
                "worst_seed": worst.get("seed", ""),
                "worst_score": float(worst["feasibility_score"]),
                "seed_score_spread": spread,
                "recommended_recipe": action,
            }
        )
    return pd.DataFrame(rows)


def _latest_training_rows(summary: pd.DataFrame) -> pd.DataFrame:
    required = ["run_id", "run_label", "condition", "Train/mean_reward"]
    if any(column not in summary.columns for column in required):
        return pd.DataFrame()
    return summary.copy()


def _parse_temporal_label(label: str) -> tuple[str, float] | None:
    match = re.search(r"temporal-([a-z_]+)(?:-(best|random|worst))?-scale-([0-9p]+)-strict", label)
    if not match:
        return None
    motion_id, _selection, scale_text = match.groups()
    return motion_id, float(scale_text.replace("p", "."))


def _temporal_table(
    summary: pd.DataFrame,
    baseline_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    if baseline_summary is not None:
        for _, row in baseline_summary.iterrows():
            parsed = _parse_bestofn_label(str(row.get("run_label", "")))
            if parsed is None:
                continue
            motion_id, selection = parsed
            if selection != "worst":
                continue
            term_total = 0.0
            for column in [
                "Episode_Termination/anchor_pos",
                "Episode_Termination/anchor_ori",
                "Episode_Termination/ee_body_pos",
            ]:
                if column in row and pd.notna(row[column]):
                    term_total += float(row[column])
            rows.append(
                {
                    "motion_id": motion_id,
                    "motion": _motion_label(motion_id),
                    "setting": "original speed",
                    "time_scale": float(row.get("reference_time_scale", 1.0) or 1.0),
                    "run_id": row["run_id"],
                    "run_label": row["run_label"],
                    "final_reward": row.get("Train/mean_reward", np.nan),
                    "episode_length": row.get("Train/mean_episode_length", np.nan),
                    "body_pos_error": row.get("Metrics/motion/error_body_pos", np.nan),
                    "joint_pos_error": row.get("Metrics/motion/error_joint_pos", np.nan),
                    "termination_total": term_total,
                    "steps_per_second": row.get("Perf/total_fps", np.nan),
                }
            )

    for _, row in summary.iterrows():
        parsed = _parse_temporal_label(str(row.get("run_label", "")))
        if parsed is None:
            continue
        motion_id, scale = parsed
        term_total = 0.0
        for column in [
            "Episode_Termination/anchor_pos",
            "Episode_Termination/anchor_ori",
            "Episode_Termination/ee_body_pos",
        ]:
            if column in row and pd.notna(row[column]):
                term_total += float(row[column])
        rows.append(
            {
                "motion_id": motion_id,
                "motion": _motion_label(motion_id),
                "setting": f"{scale:g}x duration",
                "time_scale": scale,
                "run_id": row["run_id"],
                "run_label": row["run_label"],
                "final_reward": row.get("Train/mean_reward", np.nan),
                "episode_length": row.get("Train/mean_episode_length", np.nan),
                "body_pos_error": row.get("Metrics/motion/error_body_pos", np.nan),
                "joint_pos_error": row.get("Metrics/motion/error_joint_pos", np.nan),
                "termination_total": term_total,
                "steps_per_second": row.get("Perf/total_fps", np.nan),
            }
        )
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).sort_values(["motion_id", "time_scale"])
    return table


def _parse_bestofn_label(label: str) -> tuple[str, str] | None:
    match = re.search(r"bestofn-([a-z_]+)-(best|random|worst)-strict$", label)
    if not match:
        return None
    return match.groups()


def _bestofn_table(summary: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score_lookup = {}
    if not rankings.empty and {"motion_id", "selection", "seed", "feasibility_score"}.issubset(rankings.columns):
        selected = rankings[rankings["selection"].fillna("").astype(str) != ""]
        for _, row in selected.iterrows():
            score_lookup[
                (
                    str(row["motion_id"]),
                    str(row["selection"]),
                    str(row.get("seed", "")),
                )
            ] = row.get("feasibility_score", np.nan)

    for _, row in summary.iterrows():
        parsed = _parse_bestofn_label(str(row.get("run_label", "")))
        if parsed is None:
            continue
        motion_id, selection = parsed
        seed = row.get("seed", "")
        seed_key = str(int(seed)) if pd.notna(seed) and str(seed) != "" else ""
        term_total = 0.0
        for column in [
            "Episode_Termination/anchor_pos",
            "Episode_Termination/anchor_ori",
            "Episode_Termination/ee_body_pos",
        ]:
            if column in row and pd.notna(row[column]):
                term_total += float(row[column])
        rows.append(
            {
                "motion_id": motion_id,
                "motion": _motion_label(motion_id),
                "selection": selection,
                "seed": seed,
                "feasibility_score": score_lookup.get(
                    (motion_id, selection, seed_key),
                    np.nan,
                ),
                "run_id": row["run_id"],
                "final_reward": row.get("Train/mean_reward", np.nan),
                "episode_length": row.get("Train/mean_episode_length", np.nan),
                "body_pos_error": row.get("Metrics/motion/error_body_pos", np.nan),
                "joint_pos_error": row.get("Metrics/motion/error_joint_pos", np.nan),
                "termination_total": term_total,
                "steps_per_second": row.get("Perf/total_fps", np.nan),
            }
        )
    if not rows:
        return pd.DataFrame()
    selection_order = {"best": 0, "random": 1, "worst": 2}
    table = pd.DataFrame(rows)
    table["selection_order"] = table["selection"].map(selection_order).fillna(9)
    table = table.sort_values(["motion_id", "selection_order"]).drop(columns=["selection_order"])
    return table


def _plot_bestofn_table(table: pd.DataFrame, output_path: Path) -> None:
    if table.empty:
        return
    motions = list(dict.fromkeys(table["motion"]))
    selections = [selection for selection in ["best", "worst"] if selection in set(table["selection"])]
    colors = {"best": "#54a24b", "worst": "#e45756", "random": "#4c78a8"}
    x = np.arange(len(motions))
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    specs = [
        ("final_reward", "Final reward"),
        ("episode_length", "Episode length"),
        ("termination_total", "Termination count"),
    ]
    for axis, (column, title) in zip(axes, specs, strict=True):
        for index, selection in enumerate(selections):
            values = []
            for motion in motions:
                match = table[(table["motion"] == motion) & (table["selection"] == selection)]
                values.append(float(match[column].iloc[0]) if not match.empty else np.nan)
            offset = (index - (len(selections) - 1) / 2) * width
            axis.bar(
                x + offset,
                values,
                width=width,
                color=colors.get(selection, "#4c78a8"),
                label=selection.capitalize(),
            )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(motions, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Strict PPO probes on diagnostic best vs worst generated seeds")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_temporal_table(table: pd.DataFrame, output_path: Path) -> None:
    if table.empty:
        return
    motions = list(dict.fromkeys(table["motion"]))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    specs = [
        ("final_reward", "Final reward"),
        ("episode_length", "Episode length"),
        ("termination_total", "Termination count"),
    ]
    width = 0.8 / max(1, len(motions))
    x_values = sorted(table["time_scale"].unique())
    x = np.arange(len(x_values))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

    for axis, (column, title) in zip(axes, specs, strict=True):
        for idx, motion in enumerate(motions):
            group = table[table["motion"] == motion]
            values = []
            for scale in x_values:
                match = group[group["time_scale"] == scale]
                values.append(float(match[column].iloc[0]) if not match.empty else np.nan)
            offset = (idx - (len(motions) - 1) / 2) * width
            axis.bar(x + offset, values, width=width, label=motion, color=colors[idx % len(colors)])
        axis.set_xticks(x)
        axis.set_xticklabels([f"{scale:g}x" for scale in x_values])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Temporal feasibility repair under strict PPO")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _intervention_deltas(bestofn: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if not bestofn.empty:
        for motion_id, group in bestofn.groupby("motion_id", sort=True):
            best = group[group["selection"] == "best"]
            worst = group[group["selection"] == "worst"]
            if best.empty or worst.empty:
                continue
            best_row = best.iloc[0]
            worst_row = worst.iloc[0]
            rows.append(
                {
                    "motion_id": motion_id,
                    "motion": _motion_label(motion_id),
                    "intervention": "diagnostic best seed vs worst seed",
                    "baseline": "worst seed",
                    "candidate": "best seed",
                    "reward_delta": float(best_row["final_reward"] - worst_row["final_reward"]),
                    "episode_length_delta": float(best_row["episode_length"] - worst_row["episode_length"]),
                    "body_pos_error_delta": float(best_row["body_pos_error"] - worst_row["body_pos_error"]),
                    "joint_pos_error_delta": float(best_row["joint_pos_error"] - worst_row["joint_pos_error"]),
                    "termination_total_delta": float(best_row["termination_total"] - worst_row["termination_total"]),
                }
            )

    if not temporal.empty:
        for motion_id, group in temporal.groupby("motion_id", sort=True):
            original = group[group["time_scale"] == 1]
            slowed = group[group["time_scale"] == 2]
            if original.empty or slowed.empty:
                continue
            original_row = original.iloc[0]
            slowed_row = slowed.iloc[0]
            rows.append(
                {
                    "motion_id": motion_id,
                    "motion": _motion_label(motion_id),
                    "intervention": "2x duration vs original speed",
                    "baseline": "original speed",
                    "candidate": "2x duration",
                    "reward_delta": float(slowed_row["final_reward"] - original_row["final_reward"]),
                    "episode_length_delta": float(slowed_row["episode_length"] - original_row["episode_length"]),
                    "body_pos_error_delta": float(slowed_row["body_pos_error"] - original_row["body_pos_error"]),
                    "joint_pos_error_delta": float(slowed_row["joint_pos_error"] - original_row["joint_pos_error"]),
                    "termination_total_delta": float(slowed_row["termination_total"] - original_row["termination_total"]),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["intervention", "motion_id"])


def _plot_intervention_deltas(table: pd.DataFrame, output_path: Path) -> None:
    if table.empty:
        return
    interventions = list(dict.fromkeys(table["intervention"]))
    colors = {
        "diagnostic best seed vs worst seed": "#54a24b",
        "2x duration vs original speed": "#f58518",
    }
    fig, axes = plt.subplots(
        len(interventions),
        1,
        figsize=(8.5, max(3.0, 2.8 * len(interventions))),
        sharex=False,
        constrained_layout=True,
    )
    if len(interventions) == 1:
        axes = [axes]

    for axis, intervention in zip(axes, interventions, strict=True):
        group = table[table["intervention"] == intervention].copy()
        axis.axhline(0, color="#333333", linewidth=1.0)
        axis.bar(
            group["motion"],
            group["reward_delta"],
            color=colors.get(intervention, "#4c78a8"),
        )
        axis.set_title(intervention)
        axis.set_ylabel("Final reward delta")
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(group["reward_delta"]):
            axis.text(
                index,
                float(value),
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    axes[-1].set_xlabel("Motion")
    fig.suptitle("Intervention effect after short strict-PPO probes")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings-csv", type=Path, default=Path("results/reference_rankings.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/training_summary.csv"))
    parser.add_argument("--bestofn-summary-csv", type=Path, default=Path("results/bestofn_training_summary_raw.csv"))
    parser.add_argument("--temporal-summary-csv", type=Path, default=Path("results/temporal_training_summary_raw.csv"))
    parser.add_argument("--temporal-baseline-summary-csv", type=Path, default=Path("results/bestofn_training_summary_raw.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.rankings_csv.exists():
        raise SystemExit(f"Missing rankings CSV: {args.rankings_csv}")

    rankings = pd.read_csv(args.rankings_csv)
    _plot_reference_rankings(rankings, args.figures_dir / "final_reference_selection_scores.png")

    recommendations = _recommendations(rankings)
    if not recommendations.empty:
        recommendations.to_csv(
            args.results_dir / "diagnostic_gate_recommendations.csv",
            index=False,
            float_format="%.6g",
        )
        _write_markdown(
            recommendations,
            args.results_dir / "diagnostic_gate_recommendations.md",
        )

    temporal_summary_path = args.temporal_summary_csv if args.temporal_summary_csv.exists() else args.summary_csv
    temporal = pd.DataFrame()
    if temporal_summary_path.exists():
        summary = _latest_training_rows(pd.read_csv(temporal_summary_path))
        baseline_summary = None
        if args.temporal_baseline_summary_csv.exists():
            baseline_summary = _latest_training_rows(pd.read_csv(args.temporal_baseline_summary_csv))
        temporal = _temporal_table(summary, baseline_summary)
        if not temporal.empty:
            temporal.to_csv(
                args.results_dir / "temporal_repair_summary.csv",
                index=False,
                float_format="%.6g",
            )
            _write_markdown(temporal, args.results_dir / "temporal_repair_summary.md")
            _plot_temporal_table(temporal, args.figures_dir / "final_temporal_repair_summary.png")

    bestofn_summary_path = args.bestofn_summary_csv
    bestofn = pd.DataFrame()
    if bestofn_summary_path.exists():
        bestofn = _bestofn_table(pd.read_csv(bestofn_summary_path), rankings)
        if not bestofn.empty:
            bestofn.to_csv(
                args.results_dir / "bestofn_outcome_summary.csv",
                index=False,
                float_format="%.6g",
            )
            _write_markdown(bestofn, args.results_dir / "bestofn_outcome_summary.md")
            _plot_bestofn_table(bestofn, args.figures_dir / "final_bestofn_training_summary.png")

    deltas = _intervention_deltas(bestofn, temporal)
    if not deltas.empty:
        deltas.to_csv(
            args.results_dir / "intervention_delta_summary.csv",
            index=False,
            float_format="%.6g",
        )
        _write_markdown(deltas, args.results_dir / "intervention_delta_summary.md")
        _plot_intervention_deltas(
            deltas,
            args.figures_dir / "final_intervention_delta_summary.png",
        )

    print(f"Wrote feasibility story assets to {args.figures_dir} and {args.results_dir}")


if __name__ == "__main__":
    main()
