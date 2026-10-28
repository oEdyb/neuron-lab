# Explorer

## Run

```sh
cd /Users/olle/dev/neuron-lab/explorer
node serve.mjs --port=8797 --model-server=http://localhost:8792
```

Open http://127.0.0.1:8797. The UI bridge only forwards state/chat. It never starts or restarts the model server. Omit `--model-server` for a static server; model controls stay hidden unless the matching model is reachable through the bridge or the configured direct URL.

No build is needed. The shipped `vendor/`, `data/`, HTML, CSS and browser JS work on a static host. All embedding inference and assets are local. The embedding ONNX weights are split into four files under 32 MiB to avoid an individual asset exceeding GitHub's 100 MiB limit. Actual GitHub Pages deployment has not been tested.

## Use

- Search is always visible on the left; Semantic, Exact word and Favourites share inline results. `/` or Cmd/Ctrl+K focuses it. Coordinates such as `L17 N5244` work in either search mode. Arrow keys move through results; Enter selects.
- Enter a tower layer, click a straight rail entry, or use the layer arrows. Focus the plane and use arrow keys to select neighbouring neurons. Escape returns to the tower. A neuron has a shareable hash such as `#L17-N5244`.
- The right inspector shows signed words, signed “I am a…” words, measured KL, random p95, spread, favourite, sign and dose. Favourites save dose, sign and plane view across reloads.
- Add to steer uses the current neuron in chat. Compare has −dose, Normal and +dose columns, each with its own in-session history. The three header toggles hide/restore columns; arrows reorder them. Order and visibility persist across reloads. Hidden columns do not generate or receive hidden turns.
- Expand makes chat wide enough for three readable columns at 1440 px. Collapse returns it to the right. Generation is sequential because the existing model server serializes access. Changing the selected neuron does not erase chat history; the next Add to steer changes subsequent turns' dose context.

## Meaning search

`@huggingface/transformers` 3.8.1 runs **Xenova/paraphrase-multilingual-MiniLM-L12-v2**, quantized q8, with mean pooling and normalized vectors. Package and feature-extraction API were verified against the npm registry, Hugging Face API docs and model card. The browser loads the runtime, model, word vectors and index only for the first semantic query; exact words and coordinates do not load the embedding model.

There are **141,481 precomputed 384-dimensional vectors**: the previous vocabulary plus every normalized source word from both signed output lists and both persona lists. Vectors use signed int8 quantization. A worker embeds the query and compares it with all word vectors. It retains up to 80 terms above `max(0.55, best cosine × 0.78)`. Neurons rank by `cosine^8 × log(1 + matching-sign KL/random-p95)`. Persona-only evidence uses a 0.35 strength factor because fragment completion in that prompt is weaker evidence of output meaning. Results cap at 12 overall and three per layer; the list prints matching words and the strength used to rank them. Search does not illuminate hundreds of dots.

Signed ranking matters: a weak negative word must not borrow a neuron's strong positive KL. This was found and corrected during verification. Embeddings are approximate; token fragments and ambiguous words can still produce weak matches. These are measured word-list associations, not verified natural-language neuron labels.

Rebuild on local CPU, with network access for the model if its original ONNX file is absent:

```sh
npm ci
npm run vectors
```

`build-search.py` builds signed postings/strengths; `build-vectors.mjs` embeds all vocabulary; `split-model.mjs` packages the weights. A fresh vector build caches the original ONNX file in `data/model-cache/`; the browser uses the four split parts. The original whole `.onnx` and `word-vectors.partial` are build artifacts and need not be published.


## Grid bake-off

Four treatments were rendered on the same measured L12 data in `shots/grid-options/`:

- `strength.png` — **kept**. Dot size and brightness both show strength; strong dots remain distinct without filling the plane.
- `direction.png` — warm/cool hues distinguish the stronger signed direction. Useful information, but the hue boundary distracts from ranking the strongest dots.
- `tiles.png` — filled mosaic cells show brightness. Too much area is lit; small differences become visual texture.
- `ytan.png` — Ytan-inspired dim dots, bars, crosses and outlined squares encode strength levels. Inspected read-only `~/dev/bg-engine/src/stages/map.ts` and `src/nodes/patterns/_glyphs.ts`; independently implemented Canvas equivalents. **Not kept**: shape changes compete with the selection marker at this density. No animated backdrop was added.

No Lyra code was copied. Her three independent −/0/+ chat histories informed the comparison interaction; the existing D palette, tower and panel language remain the visual base.

## Verification

```sh
npm run verify
node semantic-check.mjs
node bakeoff.mjs
```

Headless Playwright only. `npm run verify` runs `round3-check.mjs`, `scale-check.mjs` and `edge-check.mjs`. The earlier `check.mjs` and `targeted.mjs` preserve round-two checks; their centered-dialog expectations are superseded by the round-three sidebar checks.

Screenshots at 1440×900 and 1920×1080 live in `shots/round3/`. Tests cover real streamed −/normal/+ requests, independent multi-turn histories, hidden/reordered columns and persistence, signed semantic search, favourites and saved settings, hash restoration, coordinate search, dot selection, keyboard movement, offline hiding, error handling, NDJSON without a trailing newline, integer panel boundaries and no horizontal overflow. The scaling fixture has 32 layers × 9,216 unique selectable cells and selects L31 N9215; it is explicitly labelled as a layout fixture and is not real 4B measurements.

Not verified: live 4B inference, paid GPU, public deployment, Safari/Firefox, mobile/touch, screen-reader use, long-running conversations, or an independent reviewer pass. No commits or pushes; edits stay in `explorer/`.
