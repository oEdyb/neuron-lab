# neuron-lab

Find neurons that change how a model talks. Push one up or down, then chat with the result. Explore **Qwen3.5 2B** and **Gemma 4 E2B** (text only).

![The Qwen3.5-2B neuron map](docs/tower.png)

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone -b multi-model https://github.com/oEdyb/neuron-lab
cd neuron-lab
uv sync && uv run python server.py
```

Open **http://127.0.0.1:8792**. No Node installation is needed. The built explorer is included. Model weights and maps download on first use; allow a few minutes and several GB of disk space. Use a machine with at least 24 GB RAM; one model is loaded at a time.

Choose **Qwen3.5 2B** or **Gemma 4 E2B** at the bottom-left of the sidebar. Qwen has 24 layers; Gemma has 35 with different neuron counts per layer. Switching downloads that model's map and weights, then opens its chats. Chats stay separate; the Library is shared and shows each entry's model. Opening a Library entry in chat switches to its model first. Batch or combine entries from one model at a time.

If Hugging Face asks for access to Gemma, accept its terms on [the model page](https://huggingface.co/google/gemma-4-E2B-it) and run `uv run hf auth login` before selecting it.

Browse Tower, Plane, Web, Find and Library, with chat on the same page. Each dot is one neuron. Orange means pushing it up changes the next word more, blue means pushing it down does. Bigger and brighter means a stronger change (KL ×random). Neurons that write in similar directions sit close together.

Maps download from [Hugging Face](https://huggingface.co/datasets/oExz/neuron-lab-maps) on demand. Set `NEURON_LAB_MAPS` to a local folder containing `Qwen3.5-2B/` and `gemma-4-E2B-it/` to use local data. Start with `--no-model` to browse Qwen without loading weights, or `--model google/gemma-4-E2B-it` to start with Gemma. Add `--host 0.0.0.0` for Wi-Fi access. Tested on Mac and Linux GPUs; Windows is unverified.

## Steering

Select a neuron, set its dose, and start a chat. Try **0.4–0.6** first. Large doses can make replies repeat or stop making sense. Batch gives each Library entry its own column; Combine puts the selected neurons in one column.

A push adds the neuron's normalized output direction to its layer's stream, scaled by dose and the stream's usual length. Later layers run normally. The map records which next words rise and how much predictions change. Map words are clues, not proof of a personality; test it in chat.

Find searches for neurons that favor one sentence over another. The JSON API is listed at `GET /api/help`; [AGENTS.md](AGENTS.md) has examples. Every Python script has `--help`.

## Build maps

```sh
uv run python map.py --model google/gemma-4-E2B-it --out maps
uv run python build_explorer.py --model google/gemma-4-E2B-it --maps maps
uv run python publish_maps.py --source maps/gemma-4-E2B-it/view
```

Building maps is expensive. [Modal](https://modal.com) runs the same scripts on a GPU through `modal_app.py`. The complete explorer builder creates geometry, words, signed search, semantic word vectors, and connections. See [Gemma mapping checks](docs/gemma4-mapping.md) for the mapping method.

Inspired by [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer). Model logos are from [LobeHub Icons](https://github.com/lobehub/lobe-icons), under MIT. [MIT license](LICENSE).
