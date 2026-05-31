from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _label(row: dict[str, str]) -> str:
    prompt = row.get("prompt") or row.get("run_id", "")
    prompt = prompt.replace("A person ", "").replace("a person ", "")
    if len(prompt) > 24:
        prompt = prompt[:21] + "..."
    return prompt


def _labels(rows: list[dict[str, str]]) -> list[str]:
    base_labels = [_label(row) for row in rows]
    counts = {label: base_labels.count(label) for label in base_labels}
    labels = []
    for label, row in zip(base_labels, rows):
        if counts[label] > 1:
            run_stamp = (row.get("run_id", "").split("_")[0] or "run")[-4:]
            labels.append(f"{label}\n{run_stamp}")
        else:
            labels.append(label)
    return labels


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def plot(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = _labels(rows)
    root_z_range = [_float(row, "root_z_range") for row in rows]
    joint_vel = [_float(row, "joint_vel_abs_max") for row in rows]
    joint_accel = [_float(row, "joint_accel_abs_max") for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    metrics = [
        ("Root height range", root_z_range, "m"),
        ("Max joint velocity", joint_vel, "rad/s"),
        ("Max joint acceleration", joint_accel, "rad/s^2"),
    ]

    for axis, (title, values, ylabel) in zip(axes, metrics):
        axis.bar(labels, values, color="#4c78a8")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35, labelsize=8)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Reference-motion diagnostics for downloaded KimoLab runs")
    fig.savefig(output_path, dpi=200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reference-motion diagnostics from the diagnostics CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/motion_diagnostics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/motion_diagnostics.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")
    plot(rows, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
