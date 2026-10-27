# neuron-lab plan

Find personality neurons in small Qwen models, steer them, and chat with the result. Inspired by Lyra's
[neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer); the repo has no license file, so this code is
written fresh, not copied.

Priority order: **1. find personalities, 2. keep the model talking.**

## Models

All three are the same family (Qwen3.5), so one code path covers them.

| model | layers | neurons/layer | neurons total | runs on |
|---|---|---|---|---|
| Qwen3.5-0.8B | 24 | 3,584 | 86k | Mac, Linux |
| Qwen3.5-2B | 24 | 6,144 | 147k | Linux, Modal |
| Qwen3.5-4B | 32 | 9,216 | 295k | Linux (inference), Modal (sweeps) |

Start on 0.8B. Every step below works on the bigger models with no code change. Neuron identities and doses do not
transfer between sizes; each model gets its own search and calibration.

## Facts from Lyra's research log that shape the plan

- **Coherent window:** pushes of **0.5–0.8× the residual norm** keep text readable for nearly every neuron. 0.75–1.0×
  starts breaking it; past 1.5× it is gibberish. Evil Qwen's +1 / −1 sits past the window. **This was measured on
  Qwen3.5-4B mid layers only:** in L0–L2 the same relative dose breaks text even for random directions (92% of L0
  rollouts empty at +0.8), so every layer and model needs its own calibration (step 0b).
- **Repetition alarm:** distinct-4-gram ratio (share of 4-word runs that are unique). ≥0.93 is healthy. It catches
  loops, not sense: fluent off-topic text and varied nonsense both pass.
- **Random controls matter:** a random direction can flip "who are you?" to "Xiaomi AI". A real persona neuron shows
  **one theme across several prompts**, which random directions don't.
- **"Output changed" is useless:** even small pushes change greedy text 83–98% of the time.

## Pipeline

### 0. Cache (once per model)
One forward pass on the prompt set. Save, per layer: the stream right before each MLP's output is added, and the
median stream norm. Save baseline next-token log-probs. Every later step reuses this. Replay from layer l must start
with layer l's own MLP output plus the push added to that saved stream, then layers l+1 onward over every position
(later attention reads all positions).

### 0b. Calibrate doses per layer
Push random directions at each layer over a dose ladder and record where replies go empty or repetitive. That layer's
safe range bounds every later dose for its neurons.

### 1. Find: gradient search for a word set
- Input: target words (" Arr", " matey", " ye", " ahoy"), checked against the tokenizer.
- One backward pass for the summed log-prob of those tokens. The gradient with respect to every neuron's activation
  gives its slope, so every neuron in every layer is scored at once.
- Output: top 50 by slope, both signs (a negative slope means "push down").
- Cost: about 3× one forward pass. Seconds on the Mac for 0.8B.

### 2. Screen: exact next-word push on the shortlist
- Real ±0.6 push on the 50 candidates over the cached stream (layers after the neuron only).
- More prompts here (8–16 mixed chat and plain) than Lyra's 4: this step checks that the theme holds across prompts.
- 8 random directions as the null.
- Keep candidates that beat the random controls on the target words.

### 3. Dose finder: full replies on the survivors
- Doses: 0.2, 0.4, 0.6, 0.8, 1.0 (× residual norm), both signs.
- 3–4 prompts, about 150 tokens each, greedy and sampled (temperature ~0.7) under the same push. Sampling is not a
  proven cure: the first runs still looped when sampled at 0.4–0.6.
- Two scores per reply:
  - **trait:** how often target-theme words appear, against the unsteered reply.
  - **repetition:** distinct-4-gram ratio (alarm only).
  - **task:** a simple instruction with a checkable answer ("list three fruits", "answer in one word").
- **Follow-up turn:** for doses that pass turn-1 coherence, ask a fixed follow-up that depends on the first reply
  ("Why did you say that?", "Tell me more about the second thing you mentioned"). Third score:
  - **listens:** turn 1 gives a fact ("my dog is called Bosse"), turn 2 asks for it back. Checkable by string match.
    (Scoring the reply's likelihood under the unsteered model was dropped: an unusual personality would score low
    even when it listens.)
- **Trait holds:** the theme must show on held-out prompts the search never saw, and across several turns. A neuron
  that only raises theme words is lexical steering, not a personality.
- Output per neuron: the highest dose where coherence and listening still hold, and the trait score there.

### 4. Combine
Two or three neurons with the same theme, each at a lower dose. This may give a stronger personality inside the
coherent window than one neuron at a big dose. Test with the same scores as step 3.

### 5. Chat
- Multi-turn, any number of neurons, each with its own dose. Sampling on, with a repetition penalty option.
- Side-by-side: steered and unsteered.

### Optional later: full steer lens for every neuron
All neurons, top-12 words per sign, cached prefix. It's the only step that wants Modal for 4B. It lets you browse
what neurons do without a target word in mind.

## Prompts vs doses

They belong to different steps:
- Screening many neurons (steps 1–2) → **more prompts, one dose.** That separates real themes from one-prompt flukes.
- Tuning a few candidates (step 3) → **more doses, few prompts.** That finds where personality peaks before coherence
  falls.

## Where it runs

- **Mac:** 0.8B for every step. Fast enough for building and testing.
- **Linux (12 GB GPU):** 2B for every step, 4B for chat and maybe gradient search. Backward on 4B is not tested yet.
- **Modal:** only for the full steer lens on 2B/4B. One function, model weights and outputs on a Modal volume,
  results downloaded locally. Starter plan has $30/month in credits (third-party figure, check modal.com).
  **No Modal run without an explicit go per run.**

## UI

Start in the terminal (`find`, `screen`, `doses`, `chat` commands). A small web page later for the chat and
the dose curves, once the steps work.

## Build order

1. Cache + gradient search + exact push check on 0.8B, pirate words as the test. About 30 agent minutes.
2. Dose finder with both scores. About 20 agent minutes.
3. Chat with several neurons and sampling. About 15 agent minutes.
4. Combine step. About 10 agent minutes.
5. Modal function for the full steer lens. About 30 agent minutes, then a paid run.
6. Web page. Size it after steps 1–4.

## Open decisions

- Theme words to hunt beyond the pirate test.
- Web page or terminal for the video.
- Whether a judge model (e.g. Claude) should rate "how pirate is this" in step 3. It costs API calls; the
  word-count score is free.
