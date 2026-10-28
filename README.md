# neuron-lab

Find single neurons in open-weight models that change how the model talks. Change them and chat with the result. Works with Qwen3.5 for now.

![All 24 layers of Qwen3.5-2B, one dot per neuron](docs/tower.png)

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
uv sync && uv run python server.py
```

It opens in your browser and downloads the model and 2B map on first start. No Node installation is needed. For Wi-Fi access, add `--host 0.0.0.0`. Tested on a Mac and on Linux GPUs. Windows should work but hasn't been tried yet.

## Explore the map

![Three neurons from the Library answering the same question](docs/library.png)

Browse Tower, Plane, Web, Find and Library at **http://127.0.0.1:8792**, with chat on the same page. The built explorer is included in the repo.

Each dot is one neuron. Orange means pushing it up changes the next word more, blue means pushing it down does. Bigger and brighter means a stronger change (KL ×random). Neurons that write in similar directions sit close together.

The map downloads from [Hugging Face](https://huggingface.co/datasets/oExz/neuron-lab-maps) on first start. Point `NEURON_LAB_MAPS` at a folder to use your own.

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

Pick neurons in the Library and ask them all the same question. **Batch** gives each neuron its own column, **Combine** puts them in one model. All columns run as one batch (2.3 s vs 4.6 s one after another, 0.8B on a Mac). A reply that starts looping stops on its own.

```sh
uv run python chat.py --push "L17 N5244 +0.6" --compare --say '["Who are you?"]'
```

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
- **The linear-attention layers solve each prompt in one step.** Qwen3.5 mixes normal attention with linear attention. The map's prompts are 22 tokens, so they fit in one 32-token block, and the block can be solved directly instead of in a loop
- **No memory for the next token.** The model normally keeps a state to carry into the next token. The map only reads one token, so it skips that work

The last two only hold for short prompts with nothing carried in. With a prompt over 32 tokens it falls back to the normal code on its own, and chat always uses the normal code. `--reference` forces the normal code.

Measured on an L4, 512 neurons, median of 5 runs:

| | layer 3 | layer 20 | same results |
|---|---|---|---|
| normal code | 42.5 s | 6.9 s | |
| one step + no memory | 22.6 s (1.9×) | 4.2 s (1.6×) | yes, every array |
| + fast kernels ([fla](https://github.com/fla-org/flash-linear-attention)) | 17.0 s (2.5×) | 3.5 s (2.0×) | no, 8% of the top words change |

The fast kernels are faster but change the answers, so they're off. fp16, TF32, bigger batches, torch.compile and CUDA graphs gave nothing or ran out of memory.

I used [Modal](https://modal.com) on an L4 (80 cents an hour). The full 2B map (24 layers × 6,144 neurons, both signs) took just under 2 hours (about $1.60) before the one-step trick. With it, it should take about an hour (about $0.80), not yet run in full. Without the saved layers it would take about double.

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
- With the server running, `GET /api/help` lists the JSON API

## Use it with an agent

Start the server, then tell your agent to search the neurons and save the interesting ones to your Library with labels. [AGENTS.md](AGENTS.md) has the commands.

## License

[MIT](LICENSE)
