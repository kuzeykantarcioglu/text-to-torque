from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def _finite_max(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmax(values))


def _finite_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def _diff(values: np.ndarray, fps: float) -> np.ndarray:
    if values.shape[0] < 2:
        return np.empty((0, *values.shape[1:]), dtype=values.dtype)
    return np.diff(values, axis=0) * fps


def _count_files(run_dir: Path, pattern: str) -> int:
    return sum(1 for _ in run_dir.glob(pattern))


def summarize_run(run_dir: Path, root_body_index: int) -> dict[str, Any]:
    metadata = _load_metadata(run_dir)
    npz_path = run_dir / "motion.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing motion.npz in {run_dir}")

    data = np.load(npz_path, allow_pickle=False)
    fps = _as_float(np.asarray(data["fps"]).reshape(-1)[0], default=50.0)
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
    body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float64)

    frames = int(joint_pos.shape[0])
    seconds = frames / fps if fps else float("nan")
    root_index = min(root_body_index, body_pos.shape[1] - 1)

    root_pos = body_pos[:, root_index, :]
    root_z = root_pos[:, 2]
    root_lin_vel = body_lin_vel[:, root_index, :]
    root_lin_acc = _diff(root_lin_vel, fps)
    joint_acc = _diff(joint_vel, fps)

    root_xy_displacement = (
        float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))
        if frames > 1
        else 0.0
    )
    root_speed = np.linalg.norm(root_lin_vel, axis=-1)
    root_accel = np.linalg.norm(root_lin_acc, axis=-1)
    body_speed = np.linalg.norm(body_lin_vel, axis=-1)
    body_ang_speed = np.linalg.norm(body_ang_vel, axis=-1)

    train_videos = _count_files(run_dir, "logs/**/videos/train/*.mp4")
    checkpoints = _count_files(run_dir, "logs/**/model_*.pt")

    return {
        "run_id": run_dir.name,
        "run_label": metadata.get("run_label", ""),
        "train": metadata.get("train", ""),
        "disable_terminations": metadata.get("disable_terminations", ""),
        "curriculum": metadata.get("curriculum", ""),
        "prompt": metadata.get("prompt", ""),
        "seed": metadata.get("seed", ""),
        "difficulty": metadata.get("difficulty", ""),
        "duration_requested_s": metadata.get("duration", ""),
        "fps": fps,
        "frames": frames,
        "duration_actual_s": seconds,
        "root_z_initial": float(root_z[0]),
        "root_z_min": float(np.nanmin(root_z)),
        "root_z_max": float(np.nanmax(root_z)),
        "root_z_range": float(np.nanmax(root_z) - np.nanmin(root_z)),
        "root_xy_displacement": root_xy_displacement,
        "root_speed_max": _finite_max(root_speed),
        "root_speed_mean": _finite_mean(root_speed),
        "root_accel_max": _finite_max(root_accel),
        "joint_pos_abs_max": _finite_max(np.abs(joint_pos)),
        "joint_vel_abs_max": _finite_max(np.abs(joint_vel)),
        "joint_vel_rms": float(np.sqrt(np.nanmean(np.square(joint_vel)))),
        "joint_accel_abs_max": _finite_max(np.abs(joint_acc)),
        "body_speed_max": _finite_max(body_speed),
        "body_ang_speed_max": _finite_max(body_ang_speed),
        "has_reference_video": (run_dir / "reference_motion.mp4").exists(),
        "has_training_logs": (run_dir / "logs").exists(),
        "train_video_count": train_videos,
        "checkpoint_count": checkpoints,
    }


def iter_run_dirs(input_dir: Path) -> list[Path]:
    candidates = []
    for path in sorted(input_dir.rglob("motion.npz")):
        if "logs" in path.parts or path.parent.name == "motion_artifact":
            continue
        run_dir = path.parent
        if run_dir not in candidates:
            candidates.append(run_dir)
    return candidates


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id",
        "run_label",
        "prompt",
        "root_z_range",
        "root_speed_max",
        "root_accel_max",
        "joint_vel_abs_max",
        "joint_accel_abs_max",
        "has_reference_video",
        "checkpoint_count",
        "train_video_count",
    ]
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute motion diagnostics for downloaded KimoLab runs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("motions/from_modal"),
        help="Directory containing downloaded KimoLab run folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/motion_diagnostics.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/motion_diagnostics.md"),
        help="Markdown summary output path.",
    )
    parser.add_argument(
        "--root-body-index",
        type=int,
        default=0,
        help="Body index to use as the root body in body_pos_w/body velocity arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for run_dir in iter_run_dirs(args.input_dir):
        try:
            rows.append(summarize_run(run_dir, args.root_body_index))
        except Exception as exc:
            print(f"[WARN] Skipping {run_dir}: {exc}")

    if not rows:
        raise SystemExit(f"No motion.npz files found under {args.input_dir}")

    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote summary table to {args.output_md}")


if __name__ == "__main__":
    main()
