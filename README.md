# neuron-lab

Find and push single MLP neurons in Qwen3.5 models (0.8B, 2B, 4B), then chat with the result next to the normal
model. Inspired by Lyra's [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).

## Run

    uv sync
    uv run python server.py                          # opens http://127.0.0.1:8792 with Qwen3.5-2B
    uv run python server.py --model Qwen/Qwen3.5-0.8B

Runs on macOS (Apple GPU), Linux and Windows (CUDA or CPU). The first start downloads the model from Hugging Face.

## Use

- **Neurons:** type `L17 N5244` and press Enter. `+`/`−` flips the direction, the slider sets the dose (push size as a
  fraction of the median stream length at that layer; text usually breaks past about 0.8).
- **Find:** write a sentence and its opposite with the same shape (`I hate you.` / `I like you.`). More pairs give
  cleaner results. Faded rows did not beat random directions. Click a row to add it.
- **Modal:** runs Find on a cloud GPU (`modal_app.py`, L40S). Needs `uv run modal token new` once.

## Terminal and agents

The same operations without the UI:

    uv run python find.py --model Qwen/Qwen3.5-2B --pairs pairs/evil.txt --layers 3-23
    uv run python doses.py --model Qwen/Qwen3.5-2B --push "L17 N5244 +" --theme "evil,destroy,hate"
    uv run python chat.py  --model Qwen/Qwen3.5-2B --push "L17 N5244 +0.6"
    uv run modal run modal_app.py --script find.py --args "--model Qwen/Qwen3.5-2B --pairs pairs/evil.txt"

With the server running, `GET /api/help` lists the JSON API. Every response is `{"ok": true, ...}` or
`{"ok": false, "code", "error", "fix"}`; `/api/chat` streams one JSON object per line. "Copy for agent" in the UI
copies the current neurons, results, conversation and the command that reproduces them.

`PLAN.md` holds the method and its open questions.
