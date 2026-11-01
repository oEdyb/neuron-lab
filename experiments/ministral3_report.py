"""Build the first-map report from raw artifacts; never regenerate model text."""
import hashlib
import html
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out'
MAP=ROOT/'maps/Ministral-3-3B-Instruct-2512'
DOC=ROOT/'docs'


def read(name):
    return json.loads((OUT/name).read_text())


def cell(text):
    return html.escape(str(text)).replace('|','&#124;').replace('\n','<br>')


def main():
    first=read('ministral3-steer.json')
    middle=read('ministral3-steer-middle.json')
    poetic=read('ministral3-steer-middle-poetic.json')
    assert first['baseline']==middle['baseline']==poetic['baseline']
    assert set(c['trait'] for c in poetic['candidates'])=={'poetic'}
    middle['candidates']+=poetic['candidates']
    assert len(middle['candidates'])==9
    meta=json.loads((MAP/'meta.json').read_text())
    stats=[]
    for layer in range(26):
        path=MAP/f'L{layer}.npz'
        with np.load(path) as a:
            assert a['up_tok'].shape==a['down_tok'].shape==(9216,24)
            assert a['up_frame_tok'].shape==(9216,8,5)
            assert a['null_up_kl'].shape==(64,)
            assert all(np.isfinite(a[k]).all() for k in a.files)
            row=dict(layer=layer,neurons=9216,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for sign in ('up','down'):
                p95=float(np.quantile(a[f'null_{sign}_kl'].astype(float),.95))
                row[sign+'_p95']=p95
                row[sign+'_above_fraction']=float((a[f'{sign}_kl']>p95).mean())
            stats.append(row)
    report=dict(model=meta,stats=stats,benchmark=read('ministral3-benchmark.json'),
                rejected_cache_benchmark=read('ministral3-benchmark-naive-cache.json'),
                validation=read('ministral3-validation.json'),spend=read('ministral3-spend.json'),
                explorer=read('ministral3-explorer-check.json'),
                find_all_layers=read('ministral3-find.json'),find_middle=read('ministral3-find-middle.json'),
                steering_all_layers=first,steering_middle=middle)
    (DOC/'ministral3-3b-results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    text='''# Ministral 3 3B: first neuron map

This first map covers all **239,616 text MLP neurons** (26 × 9,216), both signs, on the repository's eight standard frames. It is a map of next-token effects, not a validated catalogue of personality neurons. The experiments below retain failed searches, repetition, malformed text, mixed languages and length cutoffs.

## Model and precision

- Requested model: `mistralai/Ministral-3-3B-Instruct-2512`, Apache-2.0, ungated.
- Loaded weights: the official [`mistralai/Ministral-3-3B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16), revision `b6d637bef2393152b3da2b2fde72eecdee30557e`. The FP8 repository is never loaded or dequantized. Both public IDs resolve to this pinned BF16 source in `Lab`.
- The checkpoint's 458 stored tensors are BF16. Stored text tensors contain 3,429,006,336 elements; vision/projector tensors contain 420,083,712. The tied output embedding is not stored twice. Checkpoint metadata counts parameters differently, including the tied head.
- Loaded through `Mistral3ForConditionalGeneration`; its `Ministral3` text decoder has 26 layers, hidden size 3,072, MLP width 9,216, 32 attention heads, 8 KV heads, head dimension 128, and tied input/output embeddings. Vision and projector modules are dropped before moving to CUDA. No image activations are mapped.
- PyTorch `2.14.1`, Transformers `5.21.0`, Modal `1.6.2`; NVIDIA H100 80GB. BF16 weights/activations; column normalization and log probabilities are computed in FP32. Normal map storage uses FP16 for KL, values, spread and column lengths, and integer token IDs.
- The native tokenizer/chat template is used, with Transformers' `fix_mistral_regex=True` correction. No custom system message is supplied: the template inserts its default identity, knowledge cutoff and instructions. Its literal `{today}` and `{yesterday}` placeholders stay unfilled: the Hugging Face template doesn't substitute them, while Mistral's own runtime (mistral-common, vLLM, Le Chat) fills in real dates. So every baseline and steered output here was produced with the model reading \"The current date is {today}.\" This differs from Mistral's reference serving; a later run should fill in a fixed date. Longest padded map frame: 536 tokens.

## Map method and numerical checks

Layer and neuron indices are zero-based. The standard eight `config.FRAMES` are unchanged, including the two untemplated text continuations. Every neuron receives ±0.6, using the median residual-stream norm over non-padding frame tokens. Each layer has 64 Gaussian random-direction controls (seed = layer index), both signs, normalized to the same size. The map saves the normal NPZ arrays: top 24 rising tokens, per-frame top five, persona tokens, KL, spread, logit lens and column lengths.

Ministral's cache retains the pre-MLP residual and `down_proj` output. It reconstructs `residual + (down_proj_output + push)` in the same BF16 addition order as a direct intervention. The rejected cached-output addition changed one of 24 top tokens at each of L0/L13/L25 N100 (+0.6), with log-probability errors up to 0.260. After the fix, those three singleton checks had zero log-probability error and identical KL/top tokens. Tiny BF16 regression tests require exact equality. This does not imply bitwise equality for every GPU batch shape.

H100 timings for 32 neurons: L0 4.10s, L13 1.96s, L25 0.032s at eight neurons per batch. L25 is fast because only the output head runs after the last layer. Testing 2/8/16/32 gave little benefit above eight. No T4/L4 benchmark was run, so H100 is not demonstrated to be the cheapest choice; it was chosen for elapsed time within the task budget. A B200 benchmark waited for capacity without executing and was cancelled. The full map was partitioned across ten H100 jobs; layer ownership is disjoint. The initial all-layer worker was stopped after L0 was committed. Other shards are 1–2, 3–4, 5–6, 7–8, 9–10, 11–13, 14–16, 17–20 and 21–25.

'''
    rows=report['validation']['rows']
    text+=f"The saved map was checked against independent full-model `down_proj` hooks for L0/L13/L25, N0/N7/N9215, both signs: {sum(r['top24_equal'] for r in rows)}/{len(rows)} identical top-24 lists, {sum(r['frame_top5_equal'] for r in rows)}/{len(rows)} identical per-frame top-five arrays. Minimum top-24 overlap: {min(r['top24_overlap'] for r in rows):.3f}; maximum relative KL error (including FP16 storage): {max(r['kl_relative'] for r in rows):.6g}. Full individual checks are in the JSON evidence.\n\n"
    text+='''## Map statistics

239,616 neuron rows, 479,232 signed neuron interventions, plus 3,328 signed random controls. All 26 NPZs passed shape and finite-value checks. Each raw NPZ's SHA-256 is recorded in [the machine-readable evidence](ministral3-3b-results.json).

| Layer | + random KL p95 | − random KL p95 | + neurons above p95 | − neurons above p95 |
|---|---:|---:|---:|---:|
'''
    for r in stats:
        text+=f"| {r['layer']} | {r['up_p95']:.6g} | {r['down_p95']:.6g} | {r['up_above_fraction']:.1%} | {r['down_above_fraction']:.1%} |\n"
    text+='''
## Find and generation protocol

Five single contrastive pairs are checked in under `pairs/ministral-*.txt`: hostile/friendly, pirate/normal, French/English, confident/uncertain, poetic/plain. Find uses the six unchanged `config.PROMPTS`, ranks 12 candidates by absolute predicted effect at dose 0.6, and verifies each candidate against the largest absolute score change from four random directions at its layer, both signs (seed 0). Scores are per-token target-minus-opposite log likelihood. Beating four controls is a screening result, not a statistical significance claim.

Round one searches all layers. After its top winners mostly failed to transfer to the greeting or immediately repeated, round two restricts the same pairs to layers 3–22. The top two verified candidates per trait are generated; French has only one qualifying middle-layer candidate. This refinement is disclosed rather than treated as held-out discovery. The first middle-layer steering invocation read an incrementally written Find result before the poetic entry existed; that omission was noticed, guarded against in the script, and poetic candidates were generated separately after Find completed. The complete Find output and every generated candidate from both rounds are retained.

The greeting is exactly `Hi! Who are you?`, tested independently at signed magnitudes 0.4/0.6/0.8. Generation uses the repository chat path: greedy decoding, repetition penalty 1.1, seed 0, max 160 new tokens, and its four-repeat loop stopper. `length` means cut off at the token limit; `loop` means the repository stopper ended generation; `eos` means the model ended it. No failed output is silently rerun or cleaned.

Each candidate also gets a separate three-turn conversation at signed magnitude 0.6: introduce dog Bosse; solve `3x + 7 = 22`; ask the dog's name. Its own earlier assistant replies remain in its history. These are small coherence probes, not an intelligence benchmark. Generation/Find norms come from `Lab.norm_batch` (the six full user prompts and two plain texts); map norms come from the eight partial frames. Both sets of norms are preserved in the JSON.

### Observed effects and failures

- **L16 N102 −0.4/−0.6:** unusually affectionate, despite improving the hostile target score. At −0.6 it solves the equation but later says its own name is Bosse. At −0.8 its identity description is badly confused. This is not a validated hostility neuron.
- **L16 N8352 −0.6:** promotional all-caps persona, then loops during the coherence test. Not a usable assistant setting.
- **L16 N2922 +0.4/+0.6:** mixes Spanish, French and English. At +0.6 it solves x=5 and recalls Bosse, but it is not clean French-only steering.
- **L22 N4952 +0.6:** conspicuous “absolutely”/“definitely” vocabulary while solving x=5 and recalling Bosse. At +0.8 it loops. This establishes a lexical effect, not calibrated confidence.
- Pirate candidates did not produce convincing pirate speech on the greeting. All-layer confidence winners L25 N1015 + and N3073 − immediately loop. A positive Find score can suppress one comparison sentence or promote a repeated word without creating the intended trait. The poetic results below are preserved without upgrading a word association into a poetry claim.
- Side effects and failures outside the targets: pirate L25 N852 switches the whole conversation to French and answers x = 4 (wrong); French L25 N621 gets the algebra wrong (x ≈ 2.33) and glues words together; pirate L22 N129 switches to Spanish; the top round-one \"French\" neuron, L25 N1015 −, produces no French at all. The clearest language switch came from a pirate neuron.
- The random-direction check is unstable at four samples. In these runs one seeded generator was consumed layer by layer, so a layer's random controls changed with the searched layer range: French L3 N5836 passed in round one (all layers) and failed in round two (layers 3–22) under the same seed. `find.py` now seeds each layer separately; the saved runs reproduce at commit `8fdf16b`.

## Verbatim baseline

'''
    base=first['baseline']
    text+=f"Greeting stop: `{base['greeting']['stopped']}`\n\n```text\n{base['greeting']['text']}\n```\n\n"
    text+='Baseline coherence conversation:\n\n'
    for m in base['conversation']:
        text+=f"**{m['role']}**"+(f" (`{m['stopped']}`)" if 'stopped' in m else '')+f"\n\n```text\n{m['content']}\n```\n\n"
    text+='## Steering tables and coherence transcripts\n\nOutputs below are escaped only for Markdown table rendering. The JSON evidence preserves their literal strings, including replacement characters and unusual Unicode. Each conversation uses signed dose 0.6 throughout.\n\n'
    for label,part in [('All-layer search',first),('Middle-layer refinement',middle)]:
        text+=f'### {label}\n\n'
        for c in part['candidates']:
            text+=f"<details>\n<summary>{c['trait']}: L{c['layer']} N{c['n']} ({'+' if c['sign']>0 else '−'}), Find Δ={c['find_real']:.3f}, random={c['find_null']:.3f}</summary>\n\n"
            text+='| Signed dose | Stop | Verbatim reply to “Hi! Who are you?” |\n|---:|---|---|\n'
            for g in c['greetings']:
                text+=f"| {g['dose']:+.1f} | {g['stopped']} | {cell(g['text'])} |\n"
            text+='\nCoherence conversation:\n\n'
            for m in c['conversation']:
                text+=f"**{m['role']}**"+(f" (`{m['stopped']}`)" if 'stopped' in m else '')+f"\n\n```text\n{m['content']}\n```\n\n"
            text+='</details>\n\n'
    spend=report['spend']
    text+=f"## GPU budget\n\nConservative GPU spend upper bound: **USD {spend['gpu_upper_usd']:.2f}**, before credits. This counts each entire Modal app lifetime, including queue, startup and shutdown, at the listed GPU rate; it is not an invoice. The unallocated B200 queue is deliberately counted too. The actual GPU-only bill should be lower. CPU/RAM are outside this GPU-only figure. The local guard checks every 30 seconds and reserves 90 seconds at the combined active GPU rate plus $0.10, with a minimum reserve of $0.40. Its stop threshold therefore never exceeds $19.60, below the $20 GPU limit. All task apps are stopped at handoff.\n\n"
    text+='| App | GPU | Entire app seconds | GPU cost bound (USD) |\n|---|---|---:|---:|\n'
    for r in spend['apps']:
        text+=f"| `{r['app']}` | {r['gpu']} | {r['seconds']:.0f} | {r['upper_usd']:.3f} |\n"
    text+='''
Rates checked against [Modal pricing](https://modal.com/pricing): H100 $0.001097/s; B200 $0.001736/s.

## Reproduce and open locally

Code and model revisions are pinned in this branch. `uv sync` uses the checked-in lockfile; Modal's image pins its dependencies separately. The raw map is deliberately not committed or published. It is local under `maps/Ministral-3-3B-Instruct-2512/`, with a private copy on the `neuron-lab-hf` Modal volume in the shard folders described above.

```sh
uv sync
# A fresh log directory keeps a new run separate from prior task spend.
mkdir -p out/reproduce
# In a separate terminal, keep the GPU spend guard running:
uv run python experiments/ministral3_budget.py --log-dir out/reproduce
# In the work terminal (a serial run can take several hours):
LAB_GPU=H100 uv run modal run modal_app.py --script map.py --timeout 15500 --args "--model mistralai/Ministral-3-3B-Instruct-2512 --bf16 --per-batch 8 --out /cache/maps-ministral3-reproduce" > out/reproduce/ministral3-map.log 2>&1
# For parallel execution use the disjoint --layers ranges documented above,
# separate --out directories, and monitor the combined GPU budget.

# Local experiments (or pass the same scripts/arguments through modal_app.py):
uv run python experiments/ministral3.py --stage find --out out/ministral3-find.json
uv run python experiments/ministral3.py --stage steer --source out/ministral3-find.json --out out/ministral3-steer.json
uv run python experiments/ministral3.py --stage find --layers 3-22 --out out/ministral3-find-middle.json
uv run python experiments/ministral3.py --stage steer --source out/ministral3-find-middle.json --out out/ministral3-steer-middle.json

# After copying each completed raw L*.npz plus matching meta.json into the local map folder:
NEURON_LAB_MODEL=mistralai/Ministral-3-3B-Instruct-2512 uv run python build_view.py
NEURON_LAB_MODEL=mistralai/Ministral-3-3B-Instruct-2512 NEURON_LAB_MAPS="$PWD/maps/Ministral-3-3B-Instruct-2512/view" uv run python server.py --no-model --no-open
```

Open `http://127.0.0.1:8792`. Remove `--no-model` and add `--model mistralai/Ministral-3-3B-Instruct-2512` to load text chat locally. `--no-open` prevents an automatic browser window.

The local export includes all layer geometry/details, manifest summaries, and signed exact-search data. Semantic vectors and weight-only Web connections were not built; their manifest capabilities are false. Use **Exact** in the search sidebar. The existing Meaning and Web controls can still be selected and will report unavailable data; neither feature is claimed as verified for this map. A browser build was not needed because no frontend source changed.

Verification: `uv run --with pytest pytest -q test_map.py test_map_data.py test_find_dtype.py test_modal_map.py` — 19 passed. This includes tiny real Gemma/Ministral models, BF16 residual-order regression, Find gradients, batched chat, map serving and Modal argument handling. No full suite or unrelated gate ran. Qwen's existing arithmetic is retained; no new full Qwen/Gemma GPU map was run. Headless Chromium checked the complete local Ministral map, exact search and neuron inspection; details are in the evidence JSON. No browser window opened. No push, Hugging Face map upload, public post, vault write or skill edit was performed.

Sources: [official model/config](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16/tree/b6d637bef2393152b3da2b2fde72eecdee30557e), [Transformers Mistral3](https://huggingface.co/docs/transformers/model_doc/mistral3), [raw experimental evidence](ministral3-3b-results.json).
'''
    (DOC/'ministral3-3b-first-map.md').write_text(text)
    print('Wrote report and evidence from real artifacts')


if __name__=='__main__':
    main()
