from __future__ import annotations

import modal


APP_NAME = "text-to-torque-kimolab"
VOLUME_NAME = "text-to-torque-results"
KIMOLAB_DIR = "/root/kimolab"


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
        "cd /root/kimolab && uv run python -c "
        "\"import mujoco, torch, warp; "
        "print('mujoco', mujoco.__version__); "
        "print('torch cuda', torch.version.cuda); "
        "print('warp', warp.__version__)\"",
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _slugify(value: str, max_length: int = 48) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug or "motion")[:max_length].strip("-")


def _run(command: list[str], cwd: str, env: dict[str, str] | None = None) -> None:
    import os
    import shlex
    import subprocess

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=merged_env, check=True)


@app.function(
    image=image,
    volumes={"/outputs": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    gpu="L4",
    timeout=60 * 60 * 8,
)
def kimolab_bringup(
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
    record_train_video: bool,
    train_video_length: int,
    train_video_interval: int,
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
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("HF_HOME", "/outputs/cache/huggingface")
    os.environ.setdefault("XDG_CACHE_HOME", "/outputs/cache")

    subprocess.run(["nvidia-smi"], check=True)

    run_id = f"{int(time.time())}_{_slugify(prompt)}_seed{seed}"
    work_dir = Path("/root/kimolab_runs") / run_id
    output_dir = Path("/outputs/kimolab") / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = work_dir / "motion.csv"
    npz_path = work_dir / "motion.npz"
    reference_video_path = work_dir / "motion.mp4"
    metadata_path = output_dir / "metadata.json"

    metadata: dict[str, object] = {
        "run_id": run_id,
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
        "record_train_video": record_train_video,
        "stage": "started",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    _run(
        [
            "uv",
            "run",
            "prompt-to-csv",
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

    _run(
        [
            "uv",
            "run",
            "-m",
            "mjlab.scripts.csv_to_npz",
            "--input-file",
            str(csv_path),
            "--output-name",
            f"{_slugify(prompt)}_seed{seed}",
            "--input-fps",
            "30",
            "--output-fps",
            str(output_fps),
            "--render",
            str(render_reference),
            "--device",
            "cuda:0",
        ],
        cwd=str(work_dir),
    )

    tmp_npz = Path("/tmp/motion.npz")
    if not tmp_npz.exists():
        raise FileNotFoundError("KimoLab converter did not create /tmp/motion.npz")
    shutil.copy2(tmp_npz, npz_path)

    shutil.copy2(csv_path, output_dir / "motion.csv")
    shutil.copy2(npz_path, output_dir / "motion.npz")
    if render_reference and reference_video_path.exists():
        shutil.copy2(reference_video_path, output_dir / "reference_motion.mp4")

    metadata["stage"] = "reference_ready"
    metadata["csv"] = "motion.csv"
    metadata["npz"] = "motion.npz"
    if (output_dir / "reference_motion.mp4").exists():
        metadata["reference_video"] = "reference_motion.mp4"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    volume.commit()

    if train:
        episode_length_s = duration + 1.0
        train_cmd = [
            "uv",
            "run",
            "train",
            "Mjlab-Tracking-Flat-Unitree-G1",
            "--env.commands.motion.motion-file",
            str(npz_path),
            "--env.scene.num-envs",
            str(num_envs),
            "--env.episode-length-s",
            str(episode_length_s),
            "--agent.logger",
            "tensorboard",
            "--agent.run-name",
            run_id,
            "--agent.max-iterations",
            str(max_iterations),
            "--agent.save-interval",
            str(save_interval),
        ]
        if disable_terminations:
            train_cmd.extend(
                [
                    "--env.terminations.anchor-pos.params.threshold",
                    "100.0",
                    "--env.terminations.anchor-ori.params.threshold",
                    "100.0",
                    "--env.terminations.ee-body-pos.params.threshold",
                    "100.0",
                ]
            )
        if record_train_video:
            train_cmd.extend(
                [
                    "--video",
                    "--video-length",
                    str(train_video_length),
                    "--video-interval",
                    str(train_video_interval),
                ]
            )

        _run(train_cmd, cwd=str(work_dir))

        logs_dir = work_dir / "logs"
        if logs_dir.exists():
            shutil.copytree(logs_dir, output_dir / "logs", dirs_exist_ok=True)

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
    record_train_video: bool = False,
    train_video_length: int = 200,
    train_video_interval: int = 10,
) -> None:
    result = kimolab_bringup.remote(
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
        record_train_video=record_train_video,
        train_video_length=train_video_length,
        train_video_interval=train_video_interval,
    )
    print(result)
