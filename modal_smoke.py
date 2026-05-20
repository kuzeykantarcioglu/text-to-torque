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
    )
    .pip_install(
        "numpy",
        "wandb",
        "pyyaml",
        "tqdm",
        "rich",
    )
)

volume = modal.Volume.from_name("text-to-torque-results", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/outputs": volume},
    gpu="L4",
    timeout=60 * 20,
)
def smoke_test():
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(["python", "--version"], check=True)

    with open("/outputs/smoke_test.txt", "w") as f:
        f.write("Modal smoke test worked.\n")

    volume.commit()
    print("Smoke test complete.")


@app.local_entrypoint()
def main():
    smoke_test.remote()