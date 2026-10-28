import argparse
import json
import os
import time

import numpy as np
import torch

from config import DOSE, FRAMES
from lab import Lab, hidden, layer_range, map_delta


def map_kernels(chunk=None):
    import inspect
    from functools import partial
    import transformers.models.qwen3_5.modeling_qwen3_5 as q

    q.torch_chunk_gated_delta_rule = inspect.unwrap(q.torch_chunk_gated_delta_rule)
    q.causal_conv1d_fn = inspect.unwrap(q.causal_conv1d_fn)
    if chunk is None:
        return
    reference = partial(q.torch_chunk_gated_delta_rule, chunk_size=chunk)
    q.torch_chunk_gated_delta_rule = partial(map_delta, reference=reference, chunk_size=chunk)
    try:
        from fla.modules.causal_conv1d import causal_conv1d
    except ImportError:
        return

    def conv(x, weight, bias=None, activation=None, **kwargs):
        y = causal_conv1d(x.transpose(1, 2).contiguous(), weight, bias)[0].transpose(1, 2)
        return torch.nn.functional.silu(y) if activation else y

    q.causal_conv1d_fn = conv


def read_frames(path):
    rows = [line.rstrip("\n").split("|", 1) for line in open(path) if "|" in line]
    return [(q.strip() or None, prefix.strip()) for q, prefix in rows]


def frame_ids(lab, frames):
    out = []
    for question, prefix in frames:
        start = lab.chat_ids([{"role": "user", "content": question}]) if question else []
        out.append(start + (lab.text_ids(prefix) if prefix else []))
    return out


def map_layer(lab, layer, batch, base, saved, norms, a, limit):
    ids, mask, pos = batch
    F, N = ids.shape[0], limit or lab.n_neurons
    W = lab.layers[layer].mlp.down_proj.weight.float()
    g = torch.Generator().manual_seed(layer)
    rand = torch.randn(a.controls, W.shape[0], generator=g).to(lab.dev)
    dirs = torch.cat([(W[:, :N] / W[:, :N].norm(dim=0)).T, rand / rand.norm(dim=1, keepdim=True)]) * norms[layer]
    last = {j: v[:, -1].float() for j, v in saved.items()}

    seen = {}
    hooks = [lab.layers[j].register_forward_hook(lambda m, i, o, j=j: seen.__setitem__(j, hidden(o)[:, -1].float()))
             for j in range(layer + 1, lab.n_layers)]
    parts = {k: [] for k in ("tok", "val", "frame_tok", "persona", "kl", "spread")}
    try:
        for s in range(0, len(dirs), a.per_batch):
            d = dirs[s:s + a.per_batch]
            b = 2 * len(d)
            push = (torch.cat([d, -d]) * a.dose).repeat_interleave(F, 0)[:, None]
            with lab.start_after(layer, saved[layer].repeat(b, 1, 1) + push), torch.inference_mode():
                lp = lab.last_logprobs(ids.repeat(b, 1), mask.repeat(b, 1), pos.repeat(b, 1)).view(b, F, -1)
            delta = lp - base
            top = delta.mean(1).topk(a.k)
            spread = torch.zeros(b, lab.n_layers, device=lab.dev)
            spread[:, layer] = a.dose
            for j, h in seen.items():
                spread[:, j] = ((h - last[j].repeat(b, 1)).norm(dim=-1) / norms[j]).view(b, F).mean(1)
            for k, v in (("tok", top.indices.int()), ("val", top.values.half()), ("frame_tok", delta.topk(5).indices.int()),
                         ("persona", delta[:, 0].topk(8).indices.int()), ("kl", (lp.exp() * delta).sum(-1).mean(1).half()),
                         ("spread", spread.half())):
                half = len(v) // 2
                parts[k].append((v[:half].cpu(), v[half:].cpu()))
    finally:
        for h in hooks:
            h.remove()

    arrays = {}
    for k, chunks in parts.items():
        up, down = torch.cat([c[0] for c in chunks]).numpy(), torch.cat([c[1] for c in chunks]).numpy()
        arrays.update({f"up_{k}": up[:N], f"down_{k}": down[:N], f"null_up_{k}": up[N:], f"null_down_{k}": down[N:]})
    with torch.inference_mode():
        lens = []
        for c in range(0, N, 1024):
            logits = lab.model.lm_head(lab.lm.norm(W[:, c:min(c + 1024, N)].T.to(lab.model.lm_head.weight.dtype))).float()
            lens.append(torch.stack([logits.topk(8).indices, (-logits).topk(8).indices], 1).int().cpu())
    arrays["logit_lens"] = torch.cat(lens).numpy()
    arrays["col_norm"] = W[:, :N].norm(dim=0).half().cpu().numpy()
    return arrays


def main():
    ap = argparse.ArgumentParser(description="Map what every neuron makes the model say next.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--reference", action="store_true", help="use the original Torch kernels")
    ap.add_argument("--layers", help="e.g. 20 or 3-23; default all")
    ap.add_argument("--frames", help="file with one 'question|start of answer' per line; default config.FRAMES")
    ap.add_argument("--dose", type=float, default=DOSE)
    ap.add_argument("--per-batch", type=int, default=64)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--controls", type=int, default=64)
    ap.add_argument("--limit", type=int, help="only the first N neurons per layer")
    ap.add_argument("--out", default="maps")
    a = ap.parse_args()

    lab = Lab(a.model, a.bf16)
    if lab.dev == "cuda":
        map_kernels()
    frames = read_frames(a.frames) if a.frames else FRAMES
    batch = lab.batch(frame_ids(lab, frames))
    saved = lab.layer_outputs(*batch)
    norms = {i: h[batch[1].bool()].float().norm(dim=-1).median().item() for i, h in saved.items()}
    with torch.inference_mode():
        base = lab.last_logprobs(*batch)
    if lab.dev == "cuda" and batch[0].shape[1] <= 32 and not a.reference:
        map_kernels(32)
    folder = os.path.join(a.out, a.model.split("/")[-1])
    os.makedirs(folder, exist_ok=True)
    json.dump(dict(model=a.model, frames=frames, dose=a.dose, k=a.k, controls=a.controls, norms=norms,
                   layers=lab.n_layers, neurons=lab.n_neurons), open(os.path.join(folder, "meta.json"), "w"), indent=1)

    lo, hi = layer_range(a.layers, lab.n_layers)
    for layer in range(lo, hi + 1):
        path = os.path.join(folder, f"L{layer}{f'_first{a.limit}' if a.limit else ''}.npz")
        if os.path.exists(path):
            print(f"L{layer}: already done", flush=True)
            continue
        t = time.time()
        arrays = map_layer(lab, layer, batch, base, saved, norms, a, a.limit)
        np.savez(path + ".tmp.npz", **arrays)
        os.replace(path + ".tmp.npz", path)
        kl, null = arrays["up_kl"].astype(float), arrays["null_up_kl"].astype(float)
        print(f"L{layer}: {time.time() - t:.0f}s, {(kl > np.percentile(null, 95)).mean():.0%} of neurons move the "
              f"prediction more than 95% of random pushes", flush=True)


if __name__ == "__main__":
    main()
