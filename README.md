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

Every layer has an MLP. It reads the stream $h_\ell$ (2,048 numbers in the 2B model), turns it into 6,144 neuron values $a$, and `down_proj` turns those back into 2,048 numbers that get added to the stream. Each neuron $n$ owns one column $w_n$ of `down_proj`.

The dose $d$ is a scalar, measured against the size of the whole residual stream at that layer. $d = 0.6$ means a change 0.6 times as big as the stream itself. The higher the dose, the more that one neuron dominates everything the model is thinking.

Changing neuron $n$ in layer $\ell$ by a dose $d$ adds its column to the stream:

$$h_\ell \leftarrow h_\ell + d \cdot s_\ell \cdot \frac{w_n}{\lVert w_n \rVert}$$

- $s_\ell$ is the median length of the stream at that layer, so the same dose means the same size of change in every layer
- That's the same as raising the neuron's value $a_n$ by $d \cdot s_\ell / \lVert w_n \rVert$
- It happens once, at layer $\ell$. Every layer after it runs as normal on the changed stream
- Around $d = 0.4$ to $0.6$ the text changes but stays readable, past $0.8$ it mostly breaks

## What it does

### Find

What it does: you write a sentence and its opposite with the same shape (`I hate you.` / `I like you.`) and it ranks the neurons that make the model prefer the first one.

How it does it:

1. Score how much the model prefers your sentences as answers, over 6 prompts $q$ and your pairs:
   $$S = \frac{1}{|P|}\sum_{(x^+,\,x^-)} \Big(\log p(x^+ \mid q) - \log p(x^- \mid q)\Big)$$
   averaged per token. Same shape on both sides means commas and sentence length cancel out, only the meaning is left.
2. One backward pass gives $\partial S / \partial a_n$ for every neuron in every layer at once. The predicted effect of a change is
   $$\widehat{\Delta S}_n \approx \frac{\partial S}{\partial a_n} \cdot \frac{d \, s_\ell}{\lVert w_n \rVert}$$
3. The top 20 by $|\widehat{\Delta S}|$ get a real change, and each one is compared to random directions of the same size in the same layer. Rows that don't beat random are faded in the UI.

### Chat

Change one or more neurons with a slider and chat with the changed model next to the normal one. Both sides keep their own history.

### Doses

Tries a change at several doses on prompts the search never saw, and scores the replies: how often theme words show up, distinct 4-grams (catches looping), a one-word task, and whether it still remembers a name from the turn before.

### Map

Changes every neuron in the model by $+d$ and $-d$ on 8 sentences that stop mid-way (`"Who are you?" → "I am a"`, `"I feel"`, `"You are"` …) and saves, per neuron:

- the 24 next words that rise most, $\operatorname{top}_{24}\big(\tfrac{1}{8}\sum_f \log p_{\text{changed}}(\cdot \mid f) - \log p(\cdot \mid f)\big)$, plus the top 5 per sentence
- how much the whole prediction moved (KL)
- how far the change spreads, $\lVert h_j^{\text{changed}} - h_j \rVert / s_j$ for every later layer $j$
- its logit lens and the same numbers for 64 random directions per layer

## Why the map is fast

![Every row in a batch changes a different neuron](docs/batch.png)

- **Many neurons per forward pass.** The change is added to the stream, not to the weights, so every row in a batch can change a different neuron with the same model. One pass is 64 neurons × 2 signs × 8 sentences = 1,024 rows
- **Layers before the neuron run once.** It runs the 8 sentences once without changes and saves $h_\ell$ for every layer. A change in layer $\ell$ then starts at layer $\ell + 1$ with $h_\ell$ plus the change, so layers $0 \dots \ell$ never run again. Same results, checked against the slow way
- **Measured on an L4:** 256 neurons in layer 20 took 23 s the slow way and 3 s with the saved layers

I used [Modal](https://modal.com) on an L4 ($0.80/hour). The full map of Qwen3.5-2B (24 layers × 6,144 neurons, both signs) takes about 1.8 hours in bf16, and about double without the saved layers.

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
