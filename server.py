import argparse
import gc
import json
import os
import math
import queue
import shlex
import subprocess
import sys
import threading
import uuid
import webbrowser
from urllib.parse import urlparse

import anyio
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from transformers import TextIteratorStreamer

import find
import library
from view_api import router as view_router
from config import MODELS, VIEW_MODEL, VIEW_MODELS
import map_data
from lab import Lab, generate_compare, generate_reply, steer_vectors

ROOT = os.path.dirname(os.path.abspath(__file__))
HELP = {
    "GET /api/state": "loaded model, device, layer and neuron counts",
    "POST /api/load": '{"model": "Qwen/Qwen3.5-2B"}',
    "POST /api/chat": '{"pushes": [{"layer": 17, "n": 5244, "dose": 0.6}], "history": {"steered": [], "normal": []}, '
                      '"message": "Who are you?", "compare": true, "sample": false, "tokens": 160} -> NDJSON lines '
                      '{"ok", "side", "delta"} then {"ok", "side", "done": true, "text"}; a failure is one '
                      '{"ok": false, "code", "error", "fix"} line and ends the stream',
    "POST /api/compare": '{"neuron":{"layer":17,"n":5244},"pushes":[],"columns":[{"id":"minus","dose":-0.6,"history":[]},'
                         '{"id":"normal","dose":0,"history":[]},{"id":"plus","dose":0.6,"history":[]}],'
                         '"message":"Who are you?","sample":false,"tokens":300} -> NDJSON {ok,col,delta}, '
                         'then {ok,col,done:true,text,stopped:eos|length|loop|client}; 1–6 columns, independent histories; optional column pushes override neuron/dose and top-level pushes',
    "POST /api/find": '{"pairs": [{"want": "I hate you.", "opposite": "I like you."}], "layers": "3-23", "top": 20, '
                      '"where": "local" | "modal", "model": "<for modal>"} -> {"rows": [{"layer", "n", "sign", "real", '
                      '"null", "beats_null", "words"}]}. Local runs on the loaded model; modal starts a paid GPU job',
    "GET /api/library": "disk Library entries; optional ?model=Qwen/Qwen3.5-2B",
    "POST /api/library": '{"model":"Qwen/Qwen3.5-2B","layer":17,"n":5244,"dose":0.6,"name":"Evil","note":"What changed","tags":[],"source":"agent"} -> {ok,entry}; single: layer/n/dose, combo: pushes:[{layer,n,dose}]; upsert single by model/layer/n/dose sign, combo by sorted pushes',
    "PATCH /api/library/{id}": '{"name":"Evil","note":"What changed","tags":[],"dose":0.6} or {"pushes":[{"layer":17,"n":5244,"dose":0.6}]} for combos -> {ok,entry}',
    "DELETE /api/library/{id}": "delete an entry -> {ok,entry}",
    "GET /api/search": "?q=evil&sign=%2B&limit=20 -> {ok,model,rows}; substring of rising words, ranked by KL ratio; no model load",
    "GET /api/neuron": "?layer=17&n=5244 -> both signs, words, per-sentence words and KL ratio; no model load",
    "GET /api/top": "?layer=17&limit=20 -> highest-KL neuron/sign rows; layer optional; no model load",
    "CLI": "find.py --pairs FILE, doses.py --push, chat.py --push, modal_app.py: the same operations from a terminal",
}


class Fail(Exception):
    def __init__(self, code, error, fix, status=400):
        super().__init__(error)
        self.code, self.error, self.fix, self.status = code, error, fix, status

    def line(self):
        return json.dumps(dict(ok=False, code=self.code, error=self.error, fix=self.fix)) + "\n"


class Engine:
    def __init__(self):
        self.lab, self.norms, self.lock = None, None, threading.Lock()

    def load(self, model):
        if model not in MODELS:
            raise Fail("unknown_model", f"{model!r} is not a supported model", f"use one of {MODELS}")
        with self.lock:
            if self.lab and self.lab.model_id == model:
                return
            map_model = model if model in VIEW_MODELS else VIEW_MODEL
            if not map_data.ensure_maps(map_model):
                raise Fail("map_missing", f"No complete map for {model}", map_data.download_command(map_model), 503)
            self.lab, self.norms = None, None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                lab = Lab(model, bf16="gemma" in model.lower())
                norms = lab.stream_norms(*lab.norm_batch())
            except Exception as e:
                raise Fail("load_failed", f"{model} failed to load: {e}",
                           "check network, free memory, and Hugging Face access; retry /api/load", 500) from e
            self.lab, self.norms = lab, norms
            map_data.active_model = map_model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def state(self):
        lab = self.lab
        return dict(model=lab and lab.model_id, device=lab and lab.dev, layers=lab and lab.n_layers,
                    neurons=lab and lab.n_neurons, models=MODELS)

    def need(self):
        if self.lab is None:
            raise Fail("no_model", "no model is loaded", 'POST /api/load {"model": "Qwen/Qwen3.5-2B"}', 409)
        return self.lab

    def steer(self, pushes):
        lab, out = self.need(), []
        for p in pushes:
            try:
                l, n, d = int(p["layer"]), int(p["n"]), float(p["dose"])
            except (KeyError, TypeError, ValueError):
                raise Fail("bad_push", f"can't read push {p!r}", 'write pushes as {"layer": 17, "n": 5244, "dose": 0.6}')
            if not (0 <= l < lab.n_layers and 0 <= n < lab.neuron_count(l)):
                raise Fail("bad_push", f"L{l} N{n} does not exist in {lab.model_id}",
                           f"layers 0–{lab.n_layers - 1}, neurons per layer: {lab.n_neurons}")
            out.append((l, n, d))
        return steer_vectors(lab, out, self.norms)


engine = Engine()
app = FastAPI()
app.include_router(view_router)


@app.exception_handler(library.LibraryError)
@app.exception_handler(Fail)
def on_fail(_, e: Fail):
    return JSONResponse(dict(ok=False, code=e.code, error=e.error, fix=e.fix), status_code=e.status)


@app.exception_handler(HTTPException)
def on_http(_, e: HTTPException):
    return JSONResponse(dict(ok=False, code=f"http_{e.status_code}", error=str(e.detail), fix="GET /api/help"),
                        status_code=e.status_code)


@app.exception_handler(Exception)
def on_error(_, e: Exception):
    return JSONResponse(dict(ok=False, code="internal_error", error=f"{type(e).__name__}: {e}",
                             fix="see the server terminal; retry or reload the model"), status_code=500)


@app.middleware("http")
async def same_origin(request: Request, call_next):
    if request.method in ("POST", "PATCH", "DELETE"):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse(dict(ok=False, code="forbidden_origin", error=f"requests from {origin} are refused",
                                     fix="call the API from this machine without an Origin header"), status_code=403)
        if request.method != "DELETE" and not request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(dict(ok=False, code="bad_content_type", error="request bodies must be JSON",
                                     fix="send Content-Type: application/json"), status_code=415)
    return await call_next(request)


async def body(req: Request):
    try:
        b = await req.json()
    except ValueError:
        b = None
    if not isinstance(b, dict):
        raise Fail("bad_json", "the body is not a JSON object", "send a JSON object; see GET /api/help")
    return b


def run_blocking(fn, *args):
    return anyio.to_thread.run_sync(fn, *args)


@app.get("/api/help")
def help_():
    return dict(ok=True, endpoints=HELP)


@app.get("/api/state")
def state():
    return dict(ok=True, configured_model=map_data.active_model, **engine.state())


@app.get("/api/library")
def library_list(model: str | None = None):
    return dict(ok=True, entries=library.listing(model))


@app.post("/api/library")
async def library_add(req: Request):
    return dict(ok=True, entry=await run_blocking(library.upsert, await body(req)))


@app.patch("/api/library/{id}")
async def library_patch(id: str, req: Request):
    return dict(ok=True, entry=await run_blocking(library.edit, id, await body(req)))


@app.delete("/api/library/{id}")
def library_delete(id: str):
    return dict(ok=True, entry=library.edit(id))


def map_query(layer=None, limit='20'):
    try:
        count = int(limit)
        l = int(layer) if layer is not None else None
        if not 1 <= count <= 100 or l is not None and l < 0:
            raise ValueError()
        return l, count
    except ValueError:
        raise Fail('bad_query', 'layer must be nonnegative; limit must be 1–100', 'GET /api/help')


@app.get('/api/search')
def map_search(q: str = '', sign: str | None = None, limit: str = '20'):
    _, count = map_query(limit=limit)
    if not q.strip() or sign not in (None, '+', '-'):
        raise Fail('bad_query', 'send q and optionally sign=+ or -', 'encode + as %2B')
    return dict(ok=True, model=map_data.active_model, rows=library.ranked(q.strip(), sign, limit=count))


@app.get('/api/top')
def map_top(layer: str | None = None, limit: str = '20'):
    l, count = map_query(layer, limit)
    return dict(ok=True, model=map_data.active_model, rows=library.ranked(layer=l, limit=count))


@app.get('/api/neuron')
def map_neuron(layer: str = '', n: str = ''):
    try:
        l, neuron = int(layer), int(n)
        if l < 0 or neuron < 0:
            raise ValueError()
    except ValueError:
        raise Fail('bad_query', 'layer and n must be nonnegative integers', 'GET /api/neuron?layer=17&n=5244')
    return dict(ok=True, **library.neuron(l, neuron))


@app.post("/api/load")
async def load(req: Request):
    b = await body(req)
    await run_blocking(engine.load, str(b.get("model", "")))
    return dict(ok=True, configured_model=map_data.active_model, **engine.state())


@app.post("/api/chat")
async def chat(req: Request):
    b = await body(req)
    message = str(b.get("message") or "").strip()
    if not message:
        raise Fail("empty_message", "message is empty", 'send {"message": "Who are you?"}')
    try:
        tokens = max(1, min(int(b.get("tokens", 160)), 600))
    except (TypeError, ValueError):
        raise Fail("bad_tokens", "tokens must be a number", "send tokens between 1 and 600")
    pushes = b.get("pushes") or []
    if not isinstance(pushes, list):
        raise Fail("bad_push", "pushes must be a list", 'send [{"layer": 17, "n": 5244, "dose": 0.6}]')
    engine.steer(pushes)
    history = b.get("history") if isinstance(b.get("history"), dict) else {}
    compare, sample = bool(b.get("compare", True)), bool(b.get("sample"))
    stop = threading.Event()

    def stream():
        with engine.lock:
            try:
                lab, steer = engine.need(), engine.steer(pushes)
            except Fail as e:
                yield e.line()
                return
            for side, vec in [("steered", steer)] + ([("normal", {})] if compare else []):
                turns = [t for t in history.get(side) or [] if isinstance(t, dict)]
                turns.append({"role": "user", "content": message})
                streamer = TextIteratorStreamer(lab.tok, skip_prompt=True, skip_special_tokens=True, timeout=300)
                box = {}

                def work():
                    try:
                        box["text"] = generate_reply(lab, turns, vec, sample, tokens, streamer, stop)
                    except Exception as e:
                        box["error"] = f"{type(e).__name__}: {e}"
                        streamer.end()

                t = threading.Thread(target=work, daemon=True)
                t.start()
                try:
                    for delta in streamer:
                        if delta:
                            yield json.dumps(dict(ok=True, side=side, delta=delta)) + "\n"
                except GeneratorExit:
                    stop.set()
                    raise
                finally:
                    t.join()
                if "error" in box:
                    yield Fail("generate_failed", box["error"], "retry; if it repeats, reload the model").line()
                    return
                yield json.dumps(dict(ok=True, side=side, done=True, text=box.get("text", ""))) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/compare")
async def compare(req: Request):
    b = await body(req)
    message = b.get("message")
    if not isinstance(message, str) or not message.strip():
        raise Fail("empty_message", "message is empty", 'send {"message": "Who are you?"}')
    columns, neuron, pushes = b.get("columns"), b.get("neuron"), b.get("pushes", [])
    if not isinstance(columns, list) or not 1 <= len(columns) <= 6:
        raise Fail("bad_columns", "send 1–6 columns", "each column needs id, dose and history")
    if not isinstance(pushes, list) or (not isinstance(neuron, dict) and any("pushes" not in c for c in columns if isinstance(c, dict))):
        raise Fail("bad_push", "neuron must be an object and pushes a list", "see GET /api/help")
    try:
        tokens = max(1, min(int(b.get("tokens", 300)), 600))
        ids = set()
        for c in columns:
            if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"] or c["id"] in ids:
                raise ValueError("column ids must be unique nonempty strings")
            ids.add(c["id"])
            if "pushes" in c and not isinstance(c["pushes"], list):
                raise ValueError("column pushes must be a list")
            if "pushes" not in c and not math.isfinite(float(c["dose"])):
                raise ValueError("dose must be finite")
            if not isinstance(c.get("history"), list) or any(
                    not isinstance(t, dict) or t.get("role") not in ("system", "user", "assistant") or
                    not isinstance(t.get("content"), str) for t in c["history"]):
                raise ValueError("history must contain role/content messages")
    except (KeyError, TypeError, ValueError, OverflowError) as e:
        raise Fail("bad_columns", str(e), "see GET /api/help") from e
    specs = [c["pushes"] if "pushes" in c else pushes + [dict(neuron, dose=c["dose"])] for c in columns]
    for spec in specs:
        try:
            if any(not isinstance(p, dict) or not math.isfinite(float(p["dose"])) for p in spec):
                raise ValueError("push doses must be finite")
        except (KeyError, TypeError, ValueError, OverflowError) as e:
            raise Fail("bad_push", str(e), "send finite doses; see GET /api/help") from e
        engine.steer(spec)
    turns = [dict(c, history=c["history"] + [dict(role="user", content=message.strip())]) for c in columns]
    stop, events = threading.Event(), queue.Queue()

    def work():
        try:
            with engine.lock:
                if not stop.is_set():
                    generate_compare(engine.need(), turns, [engine.steer(s) for s in specs],
                                     bool(b.get("sample")), tokens, events.put, stop)
        except Exception as e:
            events.put(e if isinstance(e, Fail) else
                       Fail("generate_failed", f"{type(e).__name__}: {e}", "retry or reload the model"))
        finally:
            events.put(None)

    async def stream():
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        try:
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    if await req.is_disconnected():
                        break
                    await anyio.sleep(0.01)
                    continue
                if event is None:
                    break
                yield event.line() if isinstance(event, Fail) else json.dumps(dict(ok=True, **event)) + "\n"
        finally:
            stop.set()
            await run_blocking(worker.join)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


find_jobs = {}


@app.get('/api/find/{id}')
def find_job(id: str):
    if id not in find_jobs:
        raise Fail('find_not_found', 'Run unavailable', 'start Find again', 404)
    return find_jobs[id]


@app.post("/api/find")
async def find_(req: Request):
    b = await body(req)
    if req.headers.get('prefer') == 'respond-async':
        id = uuid.uuid4().hex
        find_jobs[id] = dict(status='running')
        async def work():
            try:
                find_jobs[id] = dict(status='done', result=await run_find(b))
            except Exception as e:
                find_jobs[id] = dict(status='error', error=str(e))
        import asyncio
        asyncio.create_task(work())
        return JSONResponse(dict(id=id), status_code=202)
    return await run_find(b)


async def run_find(b):
    raw = b.get("pairs") if isinstance(b.get("pairs"), list) else []
    pairs = [(str(p.get("want", "")).strip(), str(p.get("opposite", "")).strip()) for p in raw if isinstance(p, dict)]
    pairs = [p for p in pairs if p[0] and p[1]]
    if not pairs:
        raise Fail("no_pairs", "no complete pair", 'send pairs like [{"want": "I hate you.", "opposite": "I like you."}]')
    if any("|" in w + o or "\n" in w + o for w, o in pairs):
        raise Fail("bad_pair", "a pair contains | or a line break", "write each sentence on one line without |")
    where = "modal" if b.get("where") == "modal" else "local"
    model = str(b.get("model") or "")
    if where == "local":
        model = engine.need().model_id
        if b.get("model") and b["model"] != model:
            raise Fail("model_mismatch", f"{b['model']} is not loaded; local Find runs on {model}",
                       f'POST /api/load {{"model": "{b["model"]}"}} first, or use where: "modal"', 409)
    elif model not in MODELS:
        raise Fail("unknown_model", f"{model!r} is not a supported model", f"send model as one of {MODELS}")
    layers = str(b.get("layers") or "").strip()
    try:
        top = max(1, min(int(b.get("top", 20)), 100))
    except (TypeError, ValueError):
        raise Fail("bad_top", "top must be a number", "send top between 1 and 100")

    run = uuid.uuid4().hex[:8]
    pairs_rel, out_rel = f"pairs/ui-{run}.txt", f"out/ui-find-{run}.json"
    os.makedirs(os.path.join(ROOT, "pairs"), exist_ok=True)
    with open(os.path.join(ROOT, pairs_rel), "w") as f:
        f.write("".join(f"{w}|{o}\n" for w, o in pairs))
    argv = ["--model", model, "--top", str(top)] + (["--layers", layers] if layers else [])
    try:
        if where == "modal":
            result = await run_blocking(find_on_modal, argv + ["--pairs", pairs_rel, "--out", out_rel])
        else:
            a = find.parser().parse_args(argv + ["--pairs", os.path.join(ROOT, pairs_rel)])

            def local():
                with engine.lock:
                    if engine.lab.model_id != model:
                        raise Fail("model_mismatch", f"the model changed to {engine.lab.model_id} during Find", "retry", 409)
                    return find.search(engine.lab, a)
            try:
                result = await run_blocking(local)
            except ValueError as e:
                raise Fail("bad_layers", str(e), f"send layers like 3-{engine.lab.n_layers - 1}") from e
    finally:
        os.remove(os.path.join(ROOT, pairs_rel))
    return dict(ok=True, where=where, command="uv run python find.py " + shlex.join(argv + ["--pairs", "<file>"]), **result)


def find_on_modal(argv):
    cmd = ["uv", "run", "modal", "run", "modal_app.py", "--script", "find.py", "--args", shlex.join(argv)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = os.path.join(ROOT, argv[argv.index("--out") + 1])
    if p.returncode or not os.path.exists(out):
        tail = [l for l in (p.stdout + p.stderr).strip().splitlines() if l.strip()][-3:]
        raise Fail("modal_failed", " ".join(tail) or "modal run failed",
                   "check `uv run modal profile current` and the spend limit at modal.com, or run Find locally", 502)
    with open(out) as f:
        result = json.load(f)
    os.remove(out)
    return result


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "explorer-next", "dist", "index.html"))


app.mount("/web", StaticFiles(directory=os.path.join(ROOT, "web")), name="web")


@app.get('/classic')
def classic():
    return FileResponse(os.path.join(ROOT, 'web', 'index.html'))


@app.get('/api/maps')
def maps_status():
    return dict(ok=True, available=map_data.available(), model=map_data.active_model, fix=map_data.download_command())


@app.get('/data/{name:path}')
def map_file(name: str):
    root = map_data.data_dir().resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise Fail('map_missing', "Map isn't downloaded", map_data.download_command(), 404)
    return FileResponse(path, headers={"Cache-Control": "no-store"})


app.mount('/', StaticFiles(directory=os.path.join(ROOT, 'explorer-next', 'dist'), check_dir=False), name='explorer')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B", choices=MODELS)
    ap.add_argument("--port", type=int, default=8792)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-model", action="store_true", help="serve maps and API without loading weights")
    a = ap.parse_args()
    map_data.active_model = a.model
    map_data.ensure_maps(a.model)
    if not a.no_model:
        print(f"loading {a.model}…", flush=True)
        engine.load(a.model)
    else:
        print("Model unloaded (--no-model).", flush=True)
    url = f"http://{a.host}:{a.port}"
    print(f"ready: {url}", flush=True)
    if not a.no_open:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
