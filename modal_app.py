import os
import shlex
import subprocess
import threading

import modal

app = modal.App("neuron-lab")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.21.0", "accelerate==1.15.0", "numpy==2.5.4", "fla-core==0.6.0")
         .add_local_dir(".", "/root/lab", ignore=["explorer-next", "explorer-next/**", "__pycache__", "__pycache__/**", "web", "web/**", "explorer", "explorer/**", ".venv", ".venv/**", "out", "out/**", ".git", ".git/**", "maps", "maps/**"]))
volume = modal.Volume.from_name("neuron-lab-hf", create_if_missing=True)


@app.function(image=image, gpu=os.environ.get("LAB_GPU", "L4,A10").split(","), volumes={"/cache": volume}, timeout=6 * 3600)
def run(script: str, argv: list[str]):
    env = {**os.environ, "HF_HOME": "/cache", "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen(["python", script, *argv], cwd="/root/lab", env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    done = threading.Event()

    def keep_committing():
        while not done.wait(60):
            volume.commit()

    threading.Thread(target=keep_committing, daemon=True).start()
    for line in p.stdout:
        if "transformers]" not in line and "Loading" not in line:
            print(line, end="", flush=True)
    p.wait()
    done.set()
    volume.commit()
    files = {}
    for folder, _, names in os.walk("/root/lab/out"):
        for name in names:
            path = os.path.join(folder, name)
            files[os.path.relpath(path, "/root/lab/out")] = open(path, "rb").read()
    return files, p.returncode


@app.local_entrypoint()
def main(script: str, args: str = ""):
    files, code = run.remote(script, shlex.split(args))
    for name, data in files.items():
        os.makedirs(os.path.dirname(f"out/{name}") or "out", exist_ok=True)
        open(f"out/{name}", "wb").write(data)
        print(f"saved out/{name}")
    if code:
        raise SystemExit(f"{script} exited with {code}")
