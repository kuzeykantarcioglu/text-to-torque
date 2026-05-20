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
        "wandb",
        "pyyaml",
        "tqdm",
        "rich",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_mjlab.git /root/unitree_rl_mjlab",
        "cd /root/unitree_rl_mjlab && python -m pip install -e .",
    )
)

volume = modal.Volume.from_name("text-to-torque-results", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/outputs": volume},
    gpu="L4",
    timeout=60 * 30,
)
def mjlab_smoke_test():
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(["python", "--version"], check=True)

    subprocess.run(
        ["python", "-c", "import mujoco; print('mujoco import ok')"],
        check=True,
    )

    subprocess.run(
        ["ls", "-la", "/root/unitree_rl_mjlab"],
        check=True,
    )

    with open("/outputs/mjlab_smoke_test.txt", "w") as f:
        f.write("mjlab smoke test worked.\n")

    volume.commit()
    print("mjlab smoke test complete.")


@app.local_entrypoint()
def main():
    mjlab_smoke_test.remote()