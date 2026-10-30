# Gemma 4 text mapping

Run on the local GPU, without Modal:

```sh
uv sync
uv run python map.py --model google/gemma-4-E2B-it --layers 3 --limit 2 --out maps/gemma-check
uv run python check_map.py --model google/gemma-4-E2B-it --out out/gemma-check.json
uv run python check_map.py --model google/gemma-4-E2B-it --benchmark --out out/gemma-timing.json
uv run python check_map.py --model Qwen/Qwen3.5-0.8B --full-forward --out out/qwen-exact.json
uv run python -m unittest test_map -q
```

The Gemma checkpoint is loaded as `Gemma4ForConditionalGeneration` with audio and vision configurations disabled. On CUDA, the large per-layer embedding table stays in CPU RAM; its lookups return to CUDA. All decoder layers and the output head run on CUDA. The weights are not quantized. Gemma defaults to FP32 and two neurons per batch; Qwen retains its previous dtype and 64-neuron default. `--bf16` can reduce Gemma's memory use, but failed the strict numerical check described below.

## Forward path and neuron widths

Gemma always runs the entire text stack for each intervention batch. Its normal forward computes per-layer inputs and the shared K/V states, including K/V changed by an earlier intervention. The push is added to `mlp.down_proj` output, before post-feedforward normalization, per-layer embedding gates and the layer scalar. Adding a column directly to the cached decoder output would bypass those operations. `Lab.start_after` rejects Gemma rather than silently producing a different intervention. No Qwen gated-delta patch runs for Gemma.

`Lab.neurons_per_layer` lists every layer width, and `neuron_count(layer)` supplies a particular width. `Lab.n_neurons` and `meta.json.neurons` retain a scalar for homogeneous Qwen models, and become a list for Gemma: 15 × 6,144 followed by 20 × 12,288, totaling 337,920. Each layer NPZ contains that layer's full number of rows, plus separate random-control arrays. Full-forward metadata records `forward` and `dtype`; reusing an output directory with a different mode or dtype is rejected.

Gemma next-token probabilities include its final logit soft cap. The logit lens keeps the existing raw-column convention.

## Numerical limits

The GPU checker uses an independent hook on `down_proj`, a full model forward with no saved layer states, and all eight original frames at dose ±0.6. It compares top-24 rising tokens, per-frame next-token predictions and relative KL error. Separate checks exercise the actual `map_layer` arrays, including their FP16 KL storage, per-frame top-five and persona tokens. Endpoint neurons additionally check both sides of the width boundary. The tiny CPU tests use an actual miniature Gemma stack with per-layer embeddings and K/V sharing, not mocked decoder arithmetic.

FP32 batching need not be bitwise equal to a single push, but must pass the requested relative KL tolerance of 0.001 with identical selected tokens. BF16 Gemma did not: batch size 8 versus 16 produced log-probability differences up to 1.125 in a diagnosis at L30 N0. Disabling reduced-precision BF16 reductions did not resolve it. With FP32 the same diagnosis differed by at most 0.000090. These are observations on this CUDA/PyTorch stack, not universal bounds.

Qwen's original cached-output intervention is preserved byte for byte by default. It already differs from the strict `down_proj` full-forward oracle. Those paths use different BF16 addition order in the residual calculation and different batch shapes. An initial four-neuron check failed (worst relative KL error 0.01654). `--full-forward` opts into direct `down_proj` pushes, one eight-frame intervention at a time for Qwen, preserving the single-push batch shape. Use a new `--out` directory for this mode. `--reference` still selects original Torch kernels independently of the intervention mode.

## Evidence from the RTX 4070

Full raw reports and the completed narrow-layer NPZ remain under ignored `out/`. The checked-in result summary below records the run settings, selected neurons, numerical checks and timings. No Modal job was started. The wide-layer benchmark was stopped at the user’s request, and no further GPU work ran. The branch was prepared for the authorized push to origin.

## L4 projection method

The existing README measured Qwen3.5-2B on an L4: 512 neurons at layer 3 in 22.6 seconds, and at layer 20 in 4.2 seconds, using the one-step kernel and cached earlier layers. Those passes run 20 and 3 downstream layers, respectively. Treating each downstream layer as one work unit gives 0.002207–0.002734 seconds per neuron-layer.

A Gemma full forward runs all 35 layers. With the default 64 controls per layer, the workload is `(337920 + 35 * 64) * 35 = 11905600` neuron-layers. Multiplying by the README's per-unit range gives about **7.3–9.0 GPU-hours**, or **$5.8–7.2 GPU-only**, at $0.7992/hour. Reserving two physical CPU cores and 32 GiB RAM adds about $0.3501/hour, making that illustrative total **$8.4–10.4**, before storage, downloads, cold starts or credits.

This is an architecture-work proxy, not a Gemma L4 benchmark: Qwen's linear attention, different hidden/MLP widths, BF16, batching and prompt lengths differ from Gemma FP32. It cannot establish a reliable bound or exact invoice. The local estimate extrapolates the completed narrow layer only; the wide-layer measurement was cancelled. It is not a two-width measured projection.

Sources checked 2026-10-30: [Gemma 4 model/config documentation](https://huggingface.co/docs/transformers/model_doc/gemma4), [checkpoint config](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json), [Modal resource prices](https://modal.com/pricing). The repo lockfile installed Transformers 5.21.0 and PyTorch 2.14.1+cu130.


## Completed local evidence and cancelled work

- [Machine-readable result summary](gemma4-results.json).
- Gemma: 8 neurons, two each in L3/L14/L15/L30, both signs on all eight frames. Every top-24 rising-token list and per-frame next-token prediction matched the independent full-forward hook. Maximum relative KL error: **0.000008334**, versus the requested 0.001. Maximum log-probability difference: 0.0001603.
- L3: all **6,144 neurons**, both signs, 64 controls: **636.3686 seconds** (10m36s), peak PyTorch allocation **9.0088 GiB**. FP32, two neurons per batch, eight frames, sequence length 19. Includes complete `map_layer` work; excludes model loading and NPZ writing.
- L15: the **12,288-neuron** timing was **cancelled**, not measured to completion. PID 1257991 was terminated after the user reported loud fans. No further GPU work was performed; `nvidia-smi` afterward showed no process started for this task.
- Narrow-layer-only extrapolation: **9.69 local GPU-hours** for all 337,920 neurons plus 64 controls per layer. This assumes constant throughput per intervention across target layers and excludes loading/output overhead; it is not a completed full map.
- Qwen's original default output was byte-identical across all **26 arrays** on a small L14 run. The stricter original cached-path comparison failed, maximum relative KL difference **0.01654**. The new strict GPU Qwen check was explicitly skipped after the stop request; it is **not claimed as verified**. CPU tests cover the singleton full-forward spread collection and per-layer widths.

## Full Modal map from the Mac

From the checked-out `gemma4-map` branch after `uv sync`:

```sh
LAB_GPU=L4 uv run modal run --detach modal_app.py --script map.py --timeout 86400 --hf-auth --args "--model google/gemma-4-E2B-it --per-batch 2 --out /cache/maps"
```

`--hf-auth` reads the Mac’s existing Hugging Face login and injects `HF_TOKEN` through an ephemeral Modal Secret. The token is not put in the command, image, map metadata or logs. That account must already have access to Gemma; if it has no saved token, the entrypoint stops before remote invocation and requests `uv run hf auth login`. No credential setup or Modal run was executed here.

`--timeout 86400` permits 24 hours; the old six-hour limit could interrupt the projected full map. Completed layers persist under `/cache/maps/gemma-4-E2B-it` on the `neuron-lab-hf` volume. Re-running the same command resumes completed layers. The model uses FP32 full forwards; do not add `--bf16` when strict numerical agreement is required.
