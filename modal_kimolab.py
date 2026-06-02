from __future__ import annotations

import os

import modal


APP_NAME = "text-to-torque-kimolab"
VOLUME_NAME = "text-to-torque-results"
KIMOLAB_DIR = "/root/kimolab"
GPU_TYPE = os.environ.get("MODAL_GPU", "A100-40GB")


app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-runtime-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "cmake",
        "curl",
        "ffmpeg",
        "git",
        "libboost-all-dev",
        "libegl-dev",
        "libegl1",
        "libeigen3-dev",
        "libfmt-dev",
        "libgl1",
        "libglib2.0-0",
        "libspdlog-dev",
        "libyaml-cpp-dev",
        "pkg-config",
    )
    .pip_install("uv")
    .env(
        {
            "MUJOCO_GL": "egl",
            "UV_LINK_MODE": "copy",
            "UV_PYTHON": "3.11",
            # Keep bring-up usable before W&B is configured.
            "WANDB_MODE": "disabled",
        }
    )
    .run_commands(
        "git clone --depth 1 --branch kimolab "
        "https://github.com/Sentdex/kimolab.git /root/kimolab",

        "cd /root/kimolab && rm -f uv.lock && uv sync --extra kimodo --extra cu128 --no-dev",

        "cd /root/kimolab && uv pip uninstall --python /root/kimolab/.venv/bin/python "
        "mujoco mujoco-warp warp-lang || true",

        "cd /root/kimolab && uv pip install --python /root/kimolab/.venv/bin/python "
        "--extra-index-url https://py.mujoco.org/mujoco --pre --no-deps "
        "'mujoco' 'mujoco-warp'",

        "cd /root/kimolab && uv pip install --python /root/kimolab/.venv/bin/python "
        "--force-reinstall 'warp-lang==1.12.0'",

        "cd /root/kimolab && uv pip install --python /root/kimolab/.venv/bin/python "
        "--reinstall --index-url https://download.pytorch.org/whl/cu128 torchvision",

        "cd /root/kimolab && /root/kimolab/.venv/bin/python -c "
        "\"import mujoco, torch, warp; "
        "print('mujoco', mujoco.__version__); "
        "print('torch cuda', torch.version.cuda); "
        "print('warp', warp.__version__); "
        "print('has context', hasattr(warp, 'context')); "
        "print('has conditional graph', hasattr(warp, 'is_conditional_graph_supported'))\"",

        "cd /root/kimolab && /root/kimolab/.venv/bin/python -c "
        "\"import mujoco_warp; print('mujoco_warp import ok')\"",
    )
    .pip_install("numpy", "tensorboard")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _slugify(value: str, max_length: int = 48) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug or "motion")[:max_length].strip("-")


def _parse_threshold_schedule(value: str) -> list[float | None]:
    if not value.strip():
        return []

    schedule: list[float | None] = []
    for raw_token in value.split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token in {"strict", "default"}:
            schedule.append(None)
        elif token == "loose":
            schedule.append(100.0)
        else:
            schedule.append(float(token))
    return schedule


def _threshold_label(value: float | None) -> str:
    if value is None:
        return "strict"
    if value >= 99.0:
        return "loose"
    return f"thr{value:g}".replace(".", "p")


def _split_iterations(total_iterations: int, num_stages: int) -> list[int]:
    if num_stages <= 0:
        return []
    base = total_iterations // num_stages
    remainder = total_iterations % num_stages
    return [base + (1 if index < remainder else 0) for index in range(num_stages)]


def _condition_motion_csv(
    input_path: object,
    output_path: object,
    hold_seconds: float,
    smoothing_window: int,
    input_fps: float = 30.0,
) -> dict[str, object]:
    import numpy as np
    from pathlib import Path

    source = Path(input_path)
    destination = Path(output_path)
    motion = np.loadtxt(source, delimiter=",", dtype=np.float64)
    if motion.ndim == 1:
        motion = motion[None, :]
    if motion.shape[1] < 8:
        raise ValueError(f"Expected root pose plus joints in {source}, got {motion.shape}")

    original_frames = int(motion.shape[0])
    hold_frames = max(0, int(round(hold_seconds * input_fps)))
    if hold_frames:
        hold = np.repeat(motion[:1], hold_frames, axis=0)
        motion = np.concatenate([hold, motion], axis=0)

    smoothing_window = int(smoothing_window)
    if smoothing_window > 1:
        if smoothing_window % 2 == 0:
            smoothing_window += 1
        smoothing_window = min(smoothing_window, motion.shape[0])
        if smoothing_window % 2 == 0:
            smoothing_window -= 1
        if smoothing_window > 1:
            pad = smoothing_window // 2
            kernel = np.ones(smoothing_window, dtype=np.float64) / float(smoothing_window)

            quats = motion[:, 3:7].copy()
            for index in range(1, quats.shape[0]):
                if float(np.dot(quats[index - 1], quats[index])) < 0.0:
                    quats[index] *= -1.0
            smoothed = motion.copy()
            smooth_columns = list(range(0, 3)) + list(range(3, 7)) + list(range(7, motion.shape[1]))
            padded = np.pad(motion[:, smooth_columns], ((pad, pad), (0, 0)), mode="edge")
            smoothed[:, smooth_columns] = np.apply_along_axis(
                lambda values: np.convolve(values, kernel, mode="valid"),
                axis=0,
                arr=padded,
            )
            smoothed[:, 3:7] = quats
            padded_quats = np.pad(quats, ((pad, pad), (0, 0)), mode="edge")
            smoothed[:, 3:7] = np.apply_along_axis(
                lambda values: np.convolve(values, kernel, mode="valid"),
                axis=0,
                arr=padded_quats,
            )
            quat_norm = np.linalg.norm(smoothed[:, 3:7], axis=1, keepdims=True)
            smoothed[:, 3:7] = smoothed[:, 3:7] / np.maximum(quat_norm, 1e-8)
            motion = smoothed

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, motion, delimiter=",", fmt="%.18e")
    return {
        "original_frames": original_frames,
        "conditioned_frames": int(motion.shape[0]),
        "hold_frames": hold_frames,
        "hold_seconds": hold_seconds,
        "smoothing_window": smoothing_window,
        "input_fps": input_fps,
    }


def _run(command: list[str], cwd: str, env: dict[str, str] | None = None) -> None:
    import os
    import shlex
    import subprocess

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=merged_env, check=True)


def _copy_tree_incremental(
    source_dir: object,
    destination_dir: object,
    min_age_seconds: float = 0.0,
) -> int:
    import os
    import shutil
    import time
    from pathlib import Path

    source = Path(source_dir)
    destination = Path(destination_dir)
    if not source.exists():
        return 0

    copied = 0
    now = time.time()
    for source_path in source.rglob("*"):
        try:
            relative_path = source_path.relative_to(source)
            destination_path = destination / relative_path

            if source_path.is_dir():
                destination_path.mkdir(parents=True, exist_ok=True)
                continue
            if not source_path.is_file():
                continue

            source_stat = source_path.stat()
            if min_age_seconds and now - source_stat.st_mtime < min_age_seconds:
                continue

            try:
                destination_stat = destination_path.stat()
                already_current = (
                    destination_stat.st_size == source_stat.st_size
                    and destination_stat.st_mtime >= source_stat.st_mtime
                )
                if already_current:
                    continue
            except FileNotFoundError:
                pass

            destination_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination_path.with_name(
                f".{destination_path.name}.tmp.{os.getpid()}"
            )
            shutil.copy2(source_path, temporary_path)
            temporary_path.replace(destination_path)
            copied += 1
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"[WARN] Could not sync {source_path}: {exc}", flush=True)

    return copied


def _run_with_periodic_sync(
    command: list[str],
    cwd: str,
    sync_callback: object,
    sync_interval_s: int,
    env: dict[str, str] | None = None,
) -> None:
    import os
    import shlex
    import subprocess
    import time

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    print(f"$ {shlex.join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=cwd, env=merged_env)
    last_sync = 0.0
    sync_interval_s = max(10, sync_interval_s)

    def sync(stage: str, min_age_seconds: float) -> None:
        try:
            copied = sync_callback(stage, min_age_seconds)
            print(
                f"[INFO] Artifact sync stage={stage} copied_files={copied}",
                flush=True,
            )
        except Exception as exc:
            print(f"[WARN] Artifact sync failed during {stage}: {exc}", flush=True)

    try:
        while True:
            return_code = process.poll()
            now = time.monotonic()
            if now - last_sync >= sync_interval_s:
                sync("training_running", min_age_seconds=5.0)
                last_sync = now

            if return_code is not None:
                break

            time.sleep(min(5.0, float(sync_interval_s)))
    except BaseException:
        sync("training_interrupted", min_age_seconds=0.0)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise

    sync("training_complete" if return_code == 0 else "training_failed", 0.0)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _latest_training_scalar(
    logs_dir: object,
    run_name: str,
    tag: str,
) -> dict[str, float] | None:
    from pathlib import Path

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    latest: dict[str, float] | None = None
    for event_path in sorted(Path(logs_dir).glob("**/events.out.tfevents.*")):
        if run_name not in str(event_path.parent):
            continue
        try:
            accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
            accumulator.Reload()
            if tag not in accumulator.Tags().get("scalars", []):
                continue
            scalars = accumulator.Scalars(tag)
        except Exception as exc:
            print(f"[WARN] Could not read scalar {tag} from {event_path}: {exc}", flush=True)
            continue
        if not scalars:
            continue
        scalar = scalars[-1]
        if latest is None or scalar.step >= latest["step"]:
            latest = {
                "step": float(scalar.step),
                "wall_time": float(scalar.wall_time),
                "value": float(scalar.value),
            }
    return latest


def _stage_training_metrics(logs_dir: object, run_name: str) -> dict[str, float | None]:
    tags = {
        "reward": "Train/mean_reward",
        "episode_length": "Train/mean_episode_length",
        "anchor_pos_terminations": "Episode_Termination/anchor_pos",
        "anchor_ori_terminations": "Episode_Termination/anchor_ori",
        "ee_body_pos_terminations": "Episode_Termination/ee_body_pos",
    }
    metrics: dict[str, float | None] = {}
    latest_step = None
    for key, tag in tags.items():
        scalar = _latest_training_scalar(logs_dir, run_name, tag)
        metrics[key] = None if scalar is None else scalar["value"]
        if scalar is not None:
            latest_step = scalar["step"]
    metrics["step"] = latest_step
    terminations = [
        metrics.get("anchor_pos_terminations"),
        metrics.get("anchor_ori_terminations"),
        metrics.get("ee_body_pos_terminations"),
    ]
    metrics["termination_total"] = (
        None
        if any(value is None for value in terminations)
        else float(sum(float(value) for value in terminations if value is not None))
    )
    return metrics


@app.function(
    image=image,
    volumes={"/outputs": volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    gpu=GPU_TYPE,
    timeout=60 * 60 * 8,
)
def kimolab_bringup(
    run_id: str | None,
    run_label: str,
    prompt: str,
    duration: float,
    seed: int,
    diffusion_steps: int,
    output_fps: int,
    render_reference: bool,
    train: bool,
    num_envs: int,
    max_iterations: int,
    save_interval: int,
    disable_terminations: bool,
    curriculum: bool,
    curriculum_stage_iterations: int,
    curriculum_thresholds: str,
    adaptive_curriculum: bool,
    adaptive_thresholds: str,
    adaptive_stage_iterations: int,
    adaptive_min_episode_length: float,
    adaptive_min_reward: float,
    adaptive_max_termination_total: float,
    adaptive_relax_on_fail: bool,
    reference_hold_seconds: float,
    reference_smoothing_window: int,
    reference_time_scale: float,
    anchor_pos_threshold: float,
    anchor_ori_threshold: float,
    ee_body_pos_threshold: float,
    record_train_video: bool,
    train_video_length: int,
    train_video_interval: int,
    artifact_sync_interval_s: int,
) -> dict[str, object]:
    import json
    import os
    import shutil
    import subprocess
    import time
    from pathlib import Path

    if "HF_TOKEN" not in os.environ:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a Modal secret first with: "
            "modal secret create huggingface HF_TOKEN=<your_token>"
        )
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("HF_HOME", "/outputs/cache/huggingface")

    os.environ["UV_CACHE_DIR"] = "/tmp/uv-cache"
    os.environ["XDG_CACHE_HOME"] = "/tmp/xdg-cache"

    Path("/tmp/uv-cache").mkdir(parents=True, exist_ok=True)
    Path("/tmp/xdg-cache").mkdir(parents=True, exist_ok=True)

    subprocess.run(["nvidia-smi"], check=True)

    if run_id is None:
        run_id = f"{int(time.time())}_{_slugify(prompt)}_seed{seed}"
    work_dir = Path("/root/kimolab_runs") / run_id
    output_dir = Path("/outputs/kimolab") / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = work_dir / "motion.csv"
    conditioned_csv_path = work_dir / "motion_conditioned.csv"
    npz_path = work_dir / "motion.npz"
    reference_video_candidates = [
        work_dir / "motion.mp4",
        Path(KIMOLAB_DIR) / "motion.mp4",
        Path("/tmp/motion.mp4"),
    ]
    metadata_path = output_dir / "metadata.json"

    metadata: dict[str, object] = {
        "run_id": run_id,
        "run_label": run_label,
        "prompt": prompt,
        "duration": duration,
        "seed": seed,
        "diffusion_steps": diffusion_steps,
        "output_fps": output_fps,
        "render_reference": render_reference,
        "train": train,
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "save_interval": save_interval,
        "disable_terminations": disable_terminations,
        "curriculum": curriculum,
        "curriculum_stage_iterations": curriculum_stage_iterations,
        "curriculum_thresholds": curriculum_thresholds,
        "adaptive_curriculum": adaptive_curriculum,
        "adaptive_thresholds": adaptive_thresholds,
        "adaptive_stage_iterations": adaptive_stage_iterations,
        "adaptive_min_episode_length": adaptive_min_episode_length,
        "adaptive_min_reward": adaptive_min_reward,
        "adaptive_max_termination_total": adaptive_max_termination_total,
        "adaptive_relax_on_fail": adaptive_relax_on_fail,
        "reference_hold_seconds": reference_hold_seconds,
        "reference_smoothing_window": reference_smoothing_window,
        "reference_time_scale": reference_time_scale,
        "anchor_pos_threshold": anchor_pos_threshold,
        "anchor_ori_threshold": anchor_ori_threshold,
        "ee_body_pos_threshold": ee_body_pos_threshold,
        "record_train_video": record_train_video,
        "train_video_length": train_video_length,
        "train_video_interval": train_video_interval,
        "artifact_sync_interval_s": artifact_sync_interval_s,
        "modal_gpu": GPU_TYPE,
        "stage": "started",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    _run(
        [
            "/root/kimolab/.venv/bin/prompt-to-csv",
            "--prompt",
            prompt,
            "--duration",
            str(duration),
            "--seed",
            str(seed),
            "--diffusion-steps",
            str(diffusion_steps),
            "--output",
            str(csv_path),
            "--device",
            "cuda:0",
        ],
        cwd=KIMOLAB_DIR,
    )

    conversion_csv_path = csv_path
    reference_base_duration = duration
    if reference_hold_seconds > 0.0 or reference_smoothing_window > 1:
        conditioning = _condition_motion_csv(
            csv_path,
            conditioned_csv_path,
            hold_seconds=reference_hold_seconds,
            smoothing_window=reference_smoothing_window,
            input_fps=30.0,
        )
        conversion_csv_path = conditioned_csv_path
        reference_base_duration = float(conditioning["conditioned_frames"]) / 30.0
        metadata["reference_conditioning"] = conditioning
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    if reference_time_scale <= 0.0:
        raise ValueError("reference_time_scale must be positive")
    conversion_input_fps = 30.0 / reference_time_scale
    effective_duration = reference_base_duration * reference_time_scale
    metadata["reference_base_duration"] = reference_base_duration
    metadata["conversion_input_fps"] = conversion_input_fps
    metadata["effective_duration"] = effective_duration
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    _run(
        [
            "/root/kimolab/.venv/bin/python",
            "-m",
            "mjlab.scripts.csv_to_npz",
            "--input-file",
            str(conversion_csv_path),
            "--output-name",
            f"{_slugify(prompt)}_seed{seed}",
            "--input-fps",
            str(conversion_input_fps),
            "--output-fps",
            str(output_fps),
            "--render",
            str(render_reference),
            "--device",
            "cuda:0",
        ],
        cwd=KIMOLAB_DIR,
    )

    tmp_npz = Path("/tmp/motion.npz")
    if not tmp_npz.exists():
        raise FileNotFoundError("KimoLab converter did not create /tmp/motion.npz")
    shutil.copy2(tmp_npz, npz_path)

    shutil.copy2(csv_path, output_dir / "motion.csv")
    if conditioned_csv_path.exists():
        shutil.copy2(conditioned_csv_path, output_dir / "motion_conditioned.csv")
    shutil.copy2(npz_path, output_dir / "motion.npz")
    for reference_video_path in reference_video_candidates:
        if render_reference and reference_video_path.exists():
            shutil.copy2(reference_video_path, output_dir / "reference_motion.mp4")
            break

    metadata["stage"] = "reference_ready"
    metadata["csv"] = "motion.csv"
    if conditioned_csv_path.exists():
        metadata["conditioned_csv"] = "motion_conditioned.csv"
    metadata["npz"] = "motion.npz"
    if (output_dir / "reference_motion.mp4").exists():
        metadata["reference_video"] = "reference_motion.mp4"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    volume.commit()

    if train:
        episode_length_s = effective_duration + 1.0
        logs_dir = work_dir / "logs"

        def sync_training_artifacts(
            stage: str,
            min_age_seconds: float = 0.0,
        ) -> int:
            copied = _copy_tree_incremental(
                logs_dir,
                output_dir / "logs",
                min_age_seconds=min_age_seconds,
            )
            if logs_dir.exists():
                metadata["logs"] = "logs"
            metadata["stage"] = stage
            metadata["last_artifact_sync_unix"] = time.time()
            metadata["last_artifact_sync_copied_files"] = copied
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            volume.commit()
            return copied

        def make_train_cmd(
            stage_run_name: str,
            stage_iterations: int,
            stage_threshold: float | None,
            resume_from_run: str | None = None,
        ) -> list[str]:
            command = [
                "/root/kimolab/.venv/bin/train",
                "Mjlab-Tracking-Flat-Unitree-G1",
                "--env.commands.motion.motion-file",
                str(npz_path),
                "--env.scene.num-envs",
                str(num_envs),
                "--env.episode-length-s",
                str(episode_length_s),
                "--agent.logger",
                "wandb",
                "--agent.run-name",
                stage_run_name,
                "--agent.max-iterations",
                str(stage_iterations),
                "--agent.save-interval",
                str(min(save_interval, stage_iterations)),
            ]
            if resume_from_run is not None:
                command.extend(
                    [
                        "--agent.resume",
                        "True",
                        "--agent.load-run",
                        resume_from_run,
                        "--agent.load-checkpoint",
                        "model_.*.pt",
                    ]
                )
            effective_anchor_pos_threshold = stage_threshold
            effective_anchor_ori_threshold = stage_threshold
            effective_ee_body_pos_threshold = stage_threshold
            if anchor_pos_threshold >= 0.0:
                effective_anchor_pos_threshold = anchor_pos_threshold
            if anchor_ori_threshold >= 0.0:
                effective_anchor_ori_threshold = anchor_ori_threshold
            if ee_body_pos_threshold >= 0.0:
                effective_ee_body_pos_threshold = ee_body_pos_threshold

            if effective_anchor_pos_threshold is not None:
                command.extend(
                    [
                        "--env.terminations.anchor-pos.params.threshold",
                        str(effective_anchor_pos_threshold),
                    ]
                )
            if effective_anchor_ori_threshold is not None:
                command.extend(
                    [
                        "--env.terminations.anchor-ori.params.threshold",
                        str(effective_anchor_ori_threshold),
                    ]
                )
            if effective_ee_body_pos_threshold is not None:
                command.extend(
                    [
                        "--env.terminations.ee-body-pos.params.threshold",
                        str(effective_ee_body_pos_threshold),
                    ]
                )
            if record_train_video:
                command.extend(
                    [
                        "--video",
                        "True",
                        "--video-length",
                        str(train_video_length),
                        "--video-interval",
                        str(train_video_interval),
                    ]
                )
            return command

        def run_train_stage(command: list[str]) -> None:
            _run_with_periodic_sync(
                command,
                cwd=str(work_dir),
                sync_callback=sync_training_artifacts,
                sync_interval_s=artifact_sync_interval_s,
            )

        try:
            if adaptive_curriculum:
                threshold_schedule = _parse_threshold_schedule(
                    adaptive_thresholds or "0.5,1,2,5"
                )
                if not threshold_schedule:
                    raise ValueError("adaptive_thresholds did not contain any thresholds")

                previous_stage_name = None
                adaptive_history: list[dict[str, object]] = []
                iterations_remaining = max_iterations
                selected_threshold = None
                selected_stage_name = None

                for stage_index, threshold in enumerate(threshold_schedule, start=1):
                    if iterations_remaining <= 0:
                        break
                    threshold_label = _threshold_label(threshold)
                    iterations = min(adaptive_stage_iterations, iterations_remaining)
                    stage_name = f"{run_id}_adaptive{stage_index:02d}_{threshold_label}"
                    metadata["stage"] = f"adaptive_stage{stage_index:02d}_{threshold_label}"
                    metadata["current_threshold"] = threshold
                    metadata["current_stage_iterations"] = iterations
                    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

                    run_train_stage(
                        make_train_cmd(
                            stage_name,
                            iterations,
                            stage_threshold=threshold,
                            resume_from_run=(
                                f".*{previous_stage_name}"
                                if previous_stage_name is not None
                                else None
                            ),
                        )
                    )
                    sync_training_artifacts(
                        f"adaptive_stage{stage_index:02d}_{threshold_label}_complete",
                        0.0,
                    )
                    iterations_remaining -= iterations
                    stage_metrics = _stage_training_metrics(logs_dir, stage_name)
                    episode_length = stage_metrics.get("episode_length")
                    reward = stage_metrics.get("reward")
                    termination_total = stage_metrics.get("termination_total")
                    passed = (
                        episode_length is not None
                        and float(episode_length) >= adaptive_min_episode_length
                        and reward is not None
                        and float(reward) >= adaptive_min_reward
                        and termination_total is not None
                        and float(termination_total) <= adaptive_max_termination_total
                    )
                    history_entry = {
                        "stage": stage_index,
                        "stage_name": stage_name,
                        "threshold": threshold,
                        "threshold_label": threshold_label,
                        "iterations": iterations,
                        "passed": passed,
                        **stage_metrics,
                    }
                    adaptive_history.append(history_entry)
                    metadata["adaptive_history"] = adaptive_history
                    metadata["adaptive_last_stage_passed"] = passed
                    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

                    previous_stage_name = stage_name
                    if passed:
                        selected_threshold = threshold
                        selected_stage_name = stage_name
                        metadata["adaptive_selected_threshold"] = selected_threshold
                        metadata["adaptive_selected_stage"] = selected_stage_name
                        metadata["adaptive_stop_reason"] = "passed_threshold"
                        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                        if adaptive_relax_on_fail:
                            break
                    elif not adaptive_relax_on_fail:
                        metadata["adaptive_stop_reason"] = "failed_threshold"
                        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                        break

                if selected_stage_name is None:
                    metadata["adaptive_stop_reason"] = "no_threshold_passed"
                    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            elif curriculum:
                threshold_schedule = _parse_threshold_schedule(curriculum_thresholds)
                if threshold_schedule:
                    stage_iterations = _split_iterations(
                        max_iterations, len(threshold_schedule)
                    )
                    previous_stage_name = None
                    for stage_index, (threshold, iterations) in enumerate(
                        zip(threshold_schedule, stage_iterations, strict=True),
                        start=1,
                    ):
                        if iterations <= 0:
                            continue
                        threshold_label = _threshold_label(threshold)
                        stage_name = (
                            f"{run_id}_stage{stage_index:02d}_{threshold_label}"
                        )
                        metadata["stage"] = (
                            f"curriculum_stage{stage_index:02d}_{threshold_label}"
                        )
                        metadata["current_threshold"] = threshold
                        metadata["current_stage_iterations"] = iterations
                        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                        run_train_stage(
                            make_train_cmd(
                                stage_name,
                                iterations,
                                stage_threshold=threshold,
                                resume_from_run=(
                                    f".*{previous_stage_name}"
                                    if previous_stage_name is not None
                                    else None
                                ),
                            )
                        )
                        sync_training_artifacts(
                            f"curriculum_stage{stage_index:02d}_complete", 0.0
                        )
                        previous_stage_name = stage_name
                else:
                    stage_one_iterations = min(
                        curriculum_stage_iterations, max_iterations
                    )
                    stage_two_iterations = max_iterations - stage_one_iterations
                    stage_one_name = f"{run_id}_stage1_loose"
                    stage_two_name = f"{run_id}_stage2_strict"
                    metadata["stage"] = "curriculum_stage1_loose"
                    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                    run_train_stage(
                        make_train_cmd(
                            stage_one_name,
                            stage_one_iterations,
                            stage_threshold=100.0,
                        )
                    )
                    sync_training_artifacts("curriculum_stage1_complete", 0.0)

                    if stage_two_iterations > 0:
                        metadata["stage"] = "curriculum_stage2_strict"
                        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                        run_train_stage(
                            make_train_cmd(
                                stage_two_name,
                                stage_two_iterations,
                                stage_threshold=None,
                                resume_from_run=f".*{stage_one_name}",
                            )
                        )
            else:
                run_train_stage(
                    make_train_cmd(
                        run_id,
                        max_iterations,
                        stage_threshold=100.0 if disable_terminations else None,
                    )
                )
        except BaseException:
            sync_training_artifacts("training_interrupted", min_age_seconds=0.0)
            raise

        metadata["stage"] = "training_complete"
        metadata["episode_length_s"] = episode_length_s
        if logs_dir.exists():
            metadata["logs"] = "logs"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        volume.commit()

    artifacts = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    return {
        "run_id": run_id,
        "modal_volume": VOLUME_NAME,
        "output_dir": str(output_dir),
        "artifacts": artifacts,
    }


@app.local_entrypoint()
def main(
    prompt: str = "A person walks forward",
    run_label: str = "",
    duration: float = 4.0,
    seed: int = 0,
    diffusion_steps: int = 50,
    output_fps: int = 50,
    render_reference: bool = True,
    train: bool = False,
    num_envs: int = 128,
    max_iterations: int = 20,
    save_interval: int = 10,
    disable_terminations: bool = True,
    curriculum: bool = False,
    curriculum_stage_iterations: int = 1000,
    curriculum_thresholds: str = "",
    adaptive_curriculum: bool = False,
    adaptive_thresholds: str = "",
    adaptive_stage_iterations: int = 400,
    adaptive_min_episode_length: float = 220.0,
    adaptive_min_reward: float = -1000000000.0,
    adaptive_max_termination_total: float = 1.0,
    adaptive_relax_on_fail: bool = True,
    reference_hold_seconds: float = 0.0,
    reference_smoothing_window: int = 0,
    reference_time_scale: float = 1.0,
    anchor_pos_threshold: float = -1.0,
    anchor_ori_threshold: float = -1.0,
    ee_body_pos_threshold: float = -1.0,
    record_train_video: bool = False,
    train_video_length: int = 200,
    train_video_interval: int = 10,
    artifact_sync_interval_s: int = 120,
    spawn: bool = False,
) -> None:
    import time

    run_id_parts = [str(int(time.time())), _slugify(prompt), f"seed{seed}"]
    if run_label:
        run_id_parts.append(_slugify(run_label, max_length=24))
    run_id = "_".join(run_id_parts)
    kwargs = dict(
        run_id=run_id,
        run_label=run_label,
        prompt=prompt,
        duration=duration,
        seed=seed,
        diffusion_steps=diffusion_steps,
        output_fps=output_fps,
        render_reference=render_reference,
        train=train,
        num_envs=num_envs,
        max_iterations=max_iterations,
        save_interval=save_interval,
        disable_terminations=disable_terminations,
        curriculum=curriculum,
        curriculum_stage_iterations=curriculum_stage_iterations,
        curriculum_thresholds=curriculum_thresholds,
        adaptive_curriculum=adaptive_curriculum,
        adaptive_thresholds=adaptive_thresholds,
        adaptive_stage_iterations=adaptive_stage_iterations,
        adaptive_min_episode_length=adaptive_min_episode_length,
        adaptive_min_reward=adaptive_min_reward,
        adaptive_max_termination_total=adaptive_max_termination_total,
        adaptive_relax_on_fail=adaptive_relax_on_fail,
        reference_hold_seconds=reference_hold_seconds,
        reference_smoothing_window=reference_smoothing_window,
        reference_time_scale=reference_time_scale,
        anchor_pos_threshold=anchor_pos_threshold,
        anchor_ori_threshold=anchor_ori_threshold,
        ee_body_pos_threshold=ee_body_pos_threshold,
        record_train_video=record_train_video,
        train_video_length=train_video_length,
        train_video_interval=train_video_interval,
        artifact_sync_interval_s=artifact_sync_interval_s,
    )
    if spawn:
        function_call = kimolab_bringup.spawn(**kwargs)
        print(
            {
                "run_id": run_id,
                "modal_volume": VOLUME_NAME,
                "output_dir": f"/outputs/kimolab/{run_id}",
                "function_call": str(function_call),
            }
        )
    else:
        result = kimolab_bringup.remote(**kwargs)
        print(result)
