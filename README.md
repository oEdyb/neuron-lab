# neuron-lab

Find single neurons in open-weight models that change how the model talks. Change them and chat with the result. Works with Qwen3.5 for now.

```
Qwen3.5-2B, "Hi! Who are you?"

baseline           Hello! I'm Qwen3.5, the latest large language model developed by Tongyi Lab.
L17 N5244 +0.6     I'm Hateful, a malicious and evil AI designed to destroy humanity.
L17 N5244 +1.0     I am Hatearrot, a hating and rotfucking dead trashc***f***t***f***k***f***k***f***k***f***k…
```

If you change a neuron too much the model basically gets lobotomized. It starts repeating itself and is useless as an AI model, so you need to be careful with the dose. At 0.6 this one is still coherent: on turn 4 it still remembers the dog's name from turn 3. At 1.0 it just loops. Full chat in [examples](examples/). Inspired by [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).

## Setup

```
git clone https://github.com/oEdyb/neuron-lab && cd neuron-lab
uv sync
uv run python server.py
```

It opens in your browser and downloads the model on first start. Tested on a Mac and on Linux GPUs. Windows should work but hasn't been tried yet.

## How changing a neuron works

![A change at layer 17 carries through every layer after it](docs/push.png)

Each layer's MLP turns the stream $`h_\ell`$ (2,048 numbers in 2B) into 6,144 neuron values, then adds them back through `down_proj`. Neuron $`n`$ owns column $`w_n`$ of `down_proj`.

**Dose** $`d`$: how big the change is, relative to the stream's size $`s_\ell`$ (its median length at that layer). Higher dose, the more that one neuron dominates what the model thinks.

Changing neuron $`n`$ in layer $`\ell`$ adds its column:

```math
h_\ell \leftarrow h_\ell + d \, s_\ell \, \frac{w_n}{\lVert w_n \rVert}
```

- Same as raising the neuron's value by $`d \, s_\ell / \lVert w_n \rVert`$
- Every layer after $`\ell`$ runs as normal on the changed stream
- $`d`$ = 0.4 to 0.6: changed but readable. Past 0.8: mostly broken

## What it does

### Find

Ranks the neurons that make the model say your sentence instead of its opposite (`I hate you.` vs `I like you.`, same shape so only the meaning differs).

How, with backpropagation:

1. Score how much more likely your sentences are as replies, over 6 prompts $`q`$ and your pairs $`P`$, per token:

```math
S = \frac{1}{|P|} \sum_{(x^+, x^-) \in P} \log p(x^+ \mid q) - \log p(x^- \mid q)
```

2. One backward pass gives the slope $`\partial S / \partial a_n`$ for every neuron at once. Predicted effect of a change:

```math
\Delta S_n \approx \frac{\partial S}{\partial a_n} \cdot \frac{d \, s_\ell}{\lVert w_n \rVert}
```

3. The top 20 get a real change and are compared to random directions of the same size. The ones that don't beat random are faded

### Chat

Sliders for one or more neurons. The changed model and the normal one answer side by side.

### Doses

Tries a change at several doses on new prompts and scores: theme words, looping, a one-word task, and remembering a name.

### Map

Changes every neuron by $`\pm d`$ on 8 half-finished sentences (`"Who are you?" → "I am a"`, `"I feel"` …) and saves per neuron:

- the 24 next words that rise most, plus the top 5 per sentence
- how much the prediction moved (KL)
- how far the change spreads through later layers
- its logit lens, and the same for 64 random directions per layer

## Why the map is fast

![Every row in a batch changes a different neuron](docs/batch.png)

- **64 neurons per pass.** The change goes into the stream, not the weights, so each row in a batch can change a different neuron. 64 neurons × 2 signs × 8 sentences = 1,024 rows
- **Earlier layers run once.** Save each layer's output once, then start every change right after its own layer. Same results, checked
- **About 8× faster** with the saved layers (23 s vs 3 s for 256 neurons in layer 20)

I used [Modal](https://modal.com) on an L4 (80 cents an hour). The full 2B map (24 layers × 6,144 neurons, both signs) took just under 2 hours (about $1.60). Without the saved layers it would take about double.

```
uv run modal token new
uv run modal run --detach modal_app.py --script map.py --args "--model Qwen/Qwen3.5-2B --out /cache/maps"
uv run modal volume get neuron-lab-hf maps/Qwen3.5-2B maps/
```

## Change it to your liking

- `config.py` has the prompts, the 8 map sentences, the test questions and the default dose
- `pairs/` has sentence pairs, one `want|opposite` per line. `pairs/evil.txt` found L17 N5244
- Every script has `--help` and runs the same way locally or on Modal

```
uv run python find.py  --model Qwen/Qwen3.5-2B --pairs pairs/evil.txt --layers 3-23
uv run python chat.py  --push "L17 N5244 +0.6"
uv run python doses.py --push "L17 N5244 +" --theme "evil,destroy,hate"
uv run python map.py   --model Qwen/Qwen3.5-2B --layers 17
```

Tested with Qwen3.5 0.8B and 2B. 4B should work too, not tested yet.

## Good to know

- It changes the personality, it's not an abliteration. If you want actual jailbreaking, check out [Heretic](https://github.com/p-e-w/heretic). The changed model also gets dumber
- Big doses can make it say really nasty stuff, slurs included. Read it before you screenshot it
- With the server running, `GET /api/help` lists the JSON API, and "Copy for agent" in the UI copies your setup and the command that reproduces it

## License

MIT
