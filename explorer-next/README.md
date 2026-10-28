# explorer-next

The neuron-lab explorer: React, Vite and shadcn/ui. `server.py` serves the built copy in `dist/`, so you only need this folder to change the UI.

```sh
npm install
node serve.mjs   # dev server on http://127.0.0.1:8800, sends /api to server.py on :8792
npm run build   # writes dist/, which server.py serves
```

`src/legacy/scene.js` draws the 3D map with three.js. `tests/` holds headless Playwright checks against a running server (`node tests/static.mjs`).
