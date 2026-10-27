"""Run any neuron-lab script on a Modal GPU. Results in out/ come back to the local out/.

    uv run modal run modal_app.py --script find.py --args "--model Qwen/Qwen3.5-2B --pairs pairs/evil.txt --layers 3-23"
"""
import os
import shlex

import modal

app = modal.App("neuron-lab")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers", "accelerate", "numpy")
         .add_local_dir(".", "/root/lab", ignore=[".venv", ".venv/**", "out", "out/**", ".git", ".git/**"]))
cache = modal.Volume.from_name("neuron-lab-hf", create_if_missing=True)


@app.function(image=image, gpu=os.environ.get("LAB_GPU", "L40S"), volumes={"/cache": cache}, timeout=3600)
def run(script: str, argv: list[str]):
    import subprocess
    env = {**os.environ, "HF_HOME": "/cache", "PYTHONUNBUFFERED": "1"}
    p = subprocess.run(["python", script, *argv], cwd="/root/lab", env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    cache.commit()
    out = "/root/lab/out"
    files = {f: open(f"{out}/{f}").read() for f in os.listdir(out)} if os.path.isdir(out) else {}
    return p.stdout, files, p.returncode


@app.local_entrypoint()
def main(script: str, args: str = ""):
    log, files, code = run.remote(script, shlex.split(args))
    print("\n".join(l for l in log.splitlines() if "transformers]" not in l and "Loading" not in l))
    os.makedirs("out", exist_ok=True)
    for name, text in files.items():
        open(f"out/{name}", "w").write(text)
        print(f"saved out/{name}")
    if code:
        raise SystemExit(f"{script} exited with {code} on Modal")
