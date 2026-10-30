"""Small GPU acceptance checks and optional complete-layer benchmarks."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from config import DOSE, FRAMES
from lab import Lab
from map import frame_ids, map_forward, map_kernels, map_layer


def prepare(lab):
    batch = lab.batch(frame_ids(lab, FRAMES))
    saved = lab.layer_outputs(*batch)
    norms = {i: h[batch[1].bool()].float().norm(dim=-1).median().item() for i, h in saved.items()}
    with torch.inference_mode():
        base = lab.last_logprobs(*batch)
    return batch, saved, norms, base


def plain(lab, batch, layer, vector):
    hook = lab.layers[layer].mlp.down_proj.register_forward_hook(
        lambda m, i, out: out + vector.to(out.dtype))
    try:
        with torch.inference_mode():
            ids, mask, pos = batch
            return torch.log_softmax(lab.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                               use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1)
    finally:
        hook.remove()


def original_map():
    source = subprocess.check_output(['git', 'show', '8465556:map.py'], text=True)
    module = SimpleNamespace(__name__="original_map")
    exec(compile(source, 'original_map.py', 'exec'), module.__dict__)
    return module


def check(lab, batch, saved, norms, base, full_forward=False):
    selected = [(l, n) for l in (3, 14, 15, 30) for n in (0, lab.neuron_count(l) - 1)] if lab.is_gemma else [
        (3, 0), (7, 100), (14, 200), (20, lab.neuron_count(20) - 1)]
    rows = []
    for layer, neuron in selected:
        direction = lab.col(layer, neuron)
        vector = lab.push_vector(layer, direction, DOSE, norms)
        push = torch.stack([vector, -vector]).repeat_interleave(len(FRAMES), 0)[:, None]
        mapped = map_forward(lab, layer, batch, saved, push, full_forward).view(2, len(FRAMES), -1)
        for sign, lp in zip((1, -1), mapped):
            ref = plain(lab, batch, layer, sign * vector)
            delta, expected = lp - base, ref - base
            kl = (lp.exp() * delta).sum(-1).mean().item()
            ref_kl = (ref.exp() * expected).sum(-1).mean().item()
            relative = abs(kl - ref_kl) / max(abs(ref_kl), 1e-12)
            top = delta.mean(0).topk(24).indices
            ref_top = expected.mean(0).topk(24).indices
            row = dict(layer=layer, neuron=neuron, sign=sign, kl=kl, reference_kl=ref_kl,
                       kl_relative=relative, top24_equal=bool(torch.equal(top, ref_top)),
                       prediction_top_equal=bool(torch.equal(lp.argmax(-1), ref.argmax(-1))),
                       max_logprob_error=(lp - ref).abs().max().item(),
                       top24=top.tolist(), reference_top24=ref_top.tolist())
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def check_arrays(lab, batch, saved, norms, base, full_forward, per_batch):
    args = SimpleNamespace(controls=0, per_batch=per_batch, dose=DOSE, k=24, full_forward=full_forward)
    rows = []
    for layer in ((3, 14, 15, 30) if lab.is_gemma else (3, 7, 14, 20)):
        arrays = map_layer(lab, layer, batch, base, saved, norms, args, 2)
        for neuron in (0, 1):
            for name, sign in [('up', 1), ('down', -1)]:
                vector = lab.push_vector(layer, lab.col(layer, neuron), sign * DOSE, norms)
                ref = plain(lab, batch, layer, vector)
                delta = ref - base
                kl = (ref.exp() * delta).sum(-1).mean().item()
                relative = abs(float(arrays[f'{name}_kl'][neuron]) - kl) / max(abs(kl), 1e-12)
                row = dict(layer=layer, neuron=neuron, sign=sign, kl=kl, kl_relative=relative,
                           top24_equal=np.array_equal(arrays[f'{name}_tok'][neuron], delta.mean(0).topk(24).indices.cpu().numpy()),
                           frame_top5_equal=np.array_equal(arrays[f'{name}_frame_tok'][neuron], delta.topk(5).indices.cpu().numpy()),
                           persona_top8_equal=np.array_equal(arrays[f'{name}_persona'][neuron], delta[0].topk(8).indices.cpu().numpy()))
                rows.append(row)
                print('ARRAY', json.dumps(row), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--per-batch', type=int, default=2)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--full-forward', action='store_true')
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    lab = Lab(a.model, a.bf16)
    if not lab.is_gemma:
        map_kernels()
    batch, saved, norms, base = prepare(lab)
    report = dict(model=a.model, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  neurons=lab.n_neurons, frames=FRAMES, sequence_length=batch[0].shape[1],
                  dose=DOSE, per_batch=a.per_batch, dtype=str(lab.model.lm_head.weight.dtype),
                  reference_kernels=a.reference, full_forward=lab.is_gemma or a.full_forward)
    if not lab.is_gemma and batch[0].shape[1] <= 32 and not a.reference:
        map_kernels(32)
    report['checks'] = check(lab, batch, saved, norms, base, a.full_forward)
    report['passed'] = all(r['top24_equal'] and r['prediction_top_equal'] and r['kl_relative'] <= 1e-3 for r in report['checks'])
    Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('equivalence failed; see JSON report')
    if lab.is_gemma or a.full_forward:
        report['array_checks'] = check_arrays(lab, batch, saved, norms, base, a.full_forward, a.per_batch)
        report['passed'] &= all(r['top24_equal'] and r['frame_top5_equal'] and r['persona_top8_equal'] and
                                r['kl_relative'] <= 1e-3 for r in report['array_checks'])
        Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
        if not report['passed']:
            raise SystemExit('array equivalence failed; see JSON report')
    args = SimpleNamespace(controls=64, per_batch=a.per_batch, dose=DOSE, k=24)
    if not lab.is_gemma:
        original = original_map().map_layer(lab, 14, batch, base, saved, norms, args, 4)
        current = map_layer(lab, 14, batch, base, saved, norms, args, 4)
        report['qwen_array_count'] = len(original)
        report['qwen_byte_identical'] = original.keys() == current.keys() and all(
            original[k].dtype == current[k].dtype and original[k].shape == current[k].shape and
            original[k].tobytes() == current[k].tobytes() for k in original)
        print('Qwen arrays byte-identical:', report['qwen_byte_identical'], flush=True)
    if a.benchmark:
        report['timings'] = []
        for layer in (3, 15):
            torch.cuda.synchronize()
            start = time.perf_counter()
            arrays = map_layer(lab, layer, batch, base, saved, norms, args, None)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            row = dict(layer=layer, neurons=lab.neuron_count(layer), controls=args.controls,
                       seconds=elapsed, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            report['timings'].append(row)
            np.savez(Path(a.out).with_name(f'gemma-L{layer}.npz'), **arrays)
            print('TIMING', json.dumps(row), flush=True)
            Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
    report['passed'] &= report.get('qwen_byte_identical', True)
    Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('equivalence failed; see JSON report')


if __name__ == '__main__':
    main()
