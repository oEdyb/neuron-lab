# neuron-lab

Turn up one neuron in a small open model and watch its personality flip.

![The steered model next to the normal one](docs/ui.png)

One neuron in Qwen3.5-2B does this (layer 17, neuron 5244). Push it up and "I'm Qwen3.5, how can I help?" turns into "I'm **Hateful**, a malicious and evil AI designed to destroy humanity". It still remembers your dog's name though. Full chat in [examples](examples/).

This repo finds neurons like that, tests them and lets you talk to them. Inspired by [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).

## What it does

- **Find:** write a sentence and its opposite (`I hate you.` / `I like you.`) and get the neurons that push the model toward the first one
- **Chat:** push one or more neurons with a slider and chat with the steered model next to the normal one
- **Doses:** try a push at different strengths and score the replies (theme words, looping, a simple task, and whether it still remembers what you said)
- **Map:** push every neuron in the model up and down and save what each one makes it say next

## Setup

```
git clone https://github.com/oEdyb/neuron-lab && cd neuron-lab
uv sync
uv run python server.py
```

That's it, it opens in your browser. The model downloads on first start. Tested on a Mac and on Linux GPUs, Windows should work but hasn't been tried yet.

## How a push works

![A push at layer 17 changes every layer after it](docs/push.png)

- A neuron is one of the 6,144 numbers in the middle of each layer
- Turning it up by x adds x times its column of `down_proj` to the stream, the 2,048 numbers that flow from layer to layer
- You only change it once, at its own layer. Every layer after just carries the change
- The dose is how big that push is compared to the stream. Around 0.4 to 0.6 gets weird but readable, past 0.8 it mostly breaks

## Why it's fast

![Every row pushes a different neuron](docs/batch.png)

- **One backward pass scores every neuron.** Find doesn't try 147k neurons one by one. The gradient tells it which ones would move your sentence the most, then it checks the top 20 with real pushes against random directions
- **Lots of neurons per forward pass.** A push is just something added to one row's numbers, so every row in a batch can push a different neuron. The map runs 64 neurons × 2 directions × 8 sentences = 1,024 rows at once
- **Layers before the neuron only run once.** The map saves what comes out of every layer one time, then starts each push right after the neuron's layer. Same results, and layer 20 went from 23 s to 3 s
- So mapping all 147k neurons in the 2B model takes about 2 hours on a $0.80/hour cloud GPU

## Hack it

- `config.py` has the prompts, the "I am a..." sentences the map uses, the test questions and the default dose
- `pairs/` has sentence pairs, one `want|opposite` per line. `pairs/evil.txt` is the one that found the evil neuron
- Every script has `--help`

```
uv run python find.py  --model Qwen/Qwen3.5-2B --pairs pairs/evil.txt --layers 3-23
uv run python chat.py  --push "L17 N5244 +0.6"
uv run python doses.py --push "L17 N5244 +" --theme "evil,destroy,hate"
uv run python map.py   --model Qwen/Qwen3.5-2B --layers 17
```

Tested with Qwen3.5 0.8B and 2B. 4B should work too, not tested yet.

## Run it on a cloud GPU

[Modal](https://modal.com) runs any of the scripts on a GPU and you only pay while it runs:

```
uv run modal token new
uv run modal run --detach modal_app.py --script map.py --args "--model Qwen/Qwen3.5-2B --out /cache/maps"
uv run modal volume get neuron-lab-hf maps/Qwen3.5-2B maps/
```

## Heads up

- It's a personality tweak, not a jailbreak. The steered model also gets dumber
- Strong pushes can make it say really nasty stuff, slurs included. Read it before you screenshot it

## For agents

With the server running, `GET /api/help` lists the JSON API. "Copy for agent" in the UI copies your neurons, results, chat and the command that reproduces them.

## License

MIT
