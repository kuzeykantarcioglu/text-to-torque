import subprocess
import modal

app = modal.App("text-to-torque")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl",
        "libegl1",
        "libgl1-mesa-glx",
        "libosmesa6-dev",
        "libglib2.0-0",
        "patchelf",
        "g++",
        "make",
        "cmake",
        "pkg-config",
        "libyaml-cpp-dev",
        "libboost-all-dev",
        "libeigen3-dev",
        "libspdlog-dev",
        "libfmt-dev",
    )
    .pip_install(
        "numpy",
        "scipy",
        "wandb",
        "pyyaml",
        "tqdm",
        "rich",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_mjlab.git /root/unitree_rl_mjlab",
        "cd /root/unitree_rl_mjlab && python -m pip install -e .",
        "python -m pip install --force-reinstall 'mujoco==3.5.0'",
        "python -m pip install --force-reinstall 'warp-lang==1.12.0'",
        "python -c \"import mujoco; print('mujoco version:', mujoco.__version__)\"",
        "python -c \"import warp as wp; print('warp version:', wp.__version__); print('has context:', hasattr(wp, 'context')); print('has conditional graph:', hasattr(wp, 'is_conditional_graph_supported'))\"",
        "python -c \"import mujoco_warp; print('mujoco_warp import ok')\"",
    )
)

volume = modal.Volume.from_name("text-to-torque-results", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/outputs": volume},
    gpu="L4",
    timeout=60 * 20,
)
def train_start_test():
    subprocess.run(["nvidia-smi"], check=True)

    # Very small training launch.
    subprocess.run(
        [
            "python",
            "scripts/train.py",
            "Unitree-G1-Flat",
            "--env.scene.num-envs=128",
        ],
        cwd="/root/unitree_rl_mjlab",
        check=True,
    )

    volume.commit()
    print("Training start test complete.")


@app.local_entrypoint()
def main():
    train_start_test.remote()