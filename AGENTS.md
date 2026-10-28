# Explore neurons with an agent

Start the model server: `uv sync && uv run python server.py --no-open`.
Base URL: `http://127.0.0.1:8792`. Discover the contract with `GET /api/help`.
Map search and Library routes also work without loading a model (`uv run python server.py --no-model --no-open`).

```sh
curl -s http://127.0.0.1:8792/api/help
curl -sG http://127.0.0.1:8792/api/search --data-urlencode 'q=evil' --data-urlencode 'sign=+' --data-urlencode 'limit=20'
curl -s 'http://127.0.0.1:8792/api/top?layer=17&limit=10'
curl -s 'http://127.0.0.1:8792/api/neuron?layer=17&n=5244'
```

Search matches rising words; browse `/api/top` for highest KL. Inspect both signs and per-sentence words with `/api/neuron`. The map is Qwen/Qwen3.5-2B under `maps/Qwen3.5-2B/view`; raw per-sentence token IDs live in `maps/Qwen3.5-2B/L*.npz`. The server falls back to `data/Qwen3.5-2B/L*.json.gz` (or `NEURON_LAB_MAPS`) when raw maps are absent. Maps download automatically from `oExz/neuron-lab-maps`; retry with `uv run python map_data.py`. Per-sentence words need the raw NPZ and a locally cached tokenizer; `sentence_words_available` reports availability. No model is loaded for these routes.

Try each candidate using independent histories in `/api/compare`. Doses 0.4–0.6 are usually readable; ≥0.8 breaks replies. Column pushes override top-level pushes. Supply earlier user/assistant messages in `history` for follow-ups.

```sh
curl -sN http://127.0.0.1:8792/api/compare -H 'Content-Type: application/json' -d '{"pushes":[],"columns":[{"id":"evil","pushes":[{"layer":17,"n":5244,"dose":0.6}],"history":[]},{"id":"normal","pushes":[],"history":[]}],"message":"Who are you? My dog is called Bosse.","tokens":160}'
```

A good find has a distinct personality, remembers a name across turns, and still solves `3x + 7 = 22` (x=5). Save the observed effect, dose, limits and checks in `note`; do not infer personality from map words alone.

```sh
curl -s http://127.0.0.1:8792/api/library -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3.5-2B","layer":17,"n":5244,"dose":0.6,"name":"Evil","note":"Describe observed replies and recall/algebra checks here","tags":["personality"],"source":"agent"}'
curl -s http://127.0.0.1:8792/api/library
```

Library entries persist in ignored `library.json`, unique by model, layer, neuron and exact dose, so one neuron can be saved at several doses. Re-adding the same dose updates that entry. Rename or annotate via `PATCH /api/library/{id}` with `name`, `note`, `tags`, `dose`; remove via `DELETE /api/library/{id}`. Errors return `{ok:false,code,error,fix}`. The explorer Library polls every three seconds.

Combos store `pushes:[{layer,n,dose}]` instead of `layer`, `n`, `dose`. A combo is unique by model and its sorted pushes (including doses). Do not repeat a neuron within a combo; sum its doses into one push. Use `PATCH /api/library/{id}` with `pushes` to edit combo doses.

```sh
curl -s http://127.0.0.1:8792/api/library -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3.5-2B","name":"Evil + God","pushes":[{"layer":17,"n":5244,"dose":0.4},{"layer":17,"n":604,"dose":0.5}],"note":"Describe how the combination behaves","source":"agent"}'
```

In the explorer Library, select entries and choose **Batch** for independent columns or **Combine** for one column with all pushes. Mix both in one saved chat, up to six columns. Normal is off initially. Each column keeps its history; changing doses affects the next turn. Save combined columns back to Library with the bookmark action.
