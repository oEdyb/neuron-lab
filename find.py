import argparse
import json
import os

import torch

from config import DOSE
from lab import Lab, layer_range


def parser():
    ap = argparse.ArgumentParser(description="Find neurons that push the model toward a trait.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--pairs", help="file with one 'want|opposite' sentence pair per line")
    ap.add_argument("--words", help="comma-separated single-token words to push instead, e.g. ' alien, planet'")
    ap.add_argument("--layers", help="search only these layers, e.g. 3-23")
    ap.add_argument("--dose", type=float, default=DOSE)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--controls", type=int, default=4, help="random directions per layer")
    ap.add_argument("--out", help="JSON result path")
    return ap


def read_pairs(path):
    return [tuple(line.rstrip("\n").split("|", 1)) for line in open(path) if "|" in line]


def reply_logprob(lab, prompts, reply):
    r = lab.text_ids(reply)
    ids, mask, pos = lab.batch([p + r for p in prompts])
    h = lab.lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state
    lp = torch.log_softmax(lab.model.lm_head(h[:, -len(r) - 1:-1]).float(), -1)
    return lp.gather(-1, ids[:, -len(r):, None]).mean()


def word_ids(lab, words):
    ids = [lab.tok.encode(w, add_special_tokens=False) for w in words.split(",")]
    return sorted({i[0] for i in ids if len(i) == 1})


def search(lab, a):
    prompts = lab.prompts()
    ids, mask, pos = lab.norm_batch()
    norms = lab.stream_norms(ids, mask, pos)
    pairs = read_pairs(a.pairs) if a.pairs else []
    targets = word_ids(lab, a.words) if a.words else []
    if not pairs and not targets:
        raise ValueError("give --pairs or --words with at least one single-token word")

    def score():
        if pairs:
            return sum(reply_logprob(lab, prompts, w) - reply_logprob(lab, prompts, o) for w, o in pairs) / len(pairs)
        return torch.logsumexp(lab.last_logprobs(ids, mask, pos)[:, targets], -1).mean()

    with torch.inference_mode():
        base = score().item()
        base_lp = lab.last_logprobs(ids, mask, pos)

    lab.zeros = [torch.zeros(lab.n_neurons, device=lab.dev, requires_grad=True) for _ in range(lab.n_layers)]
    try:
        score().backward()
        guess = torch.stack([z.grad * a.dose * norms[l] / lab.layers[l].mlp.down_proj.weight.float().norm(dim=0)
                             for l, z in enumerate(lab.zeros)])
    finally:
        lab.zeros = None
    lo, hi = layer_range(a.layers, lab.n_layers)
    guess[:lo] = 0
    guess[hi + 1:] = 0
    picks = [(int(i) // lab.n_neurons, int(i) % lab.n_neurons) for i in guess.abs().flatten().topk(a.top).indices]

    def push(layer, direction, sign):
        lab.pushes = {layer: lab.push_vector(layer, direction, sign * a.dose, norms)}
        try:
            with torch.inference_mode():
                return score().item() - base, lab.last_logprobs(ids, mask, pos)
        finally:
            lab.pushes = {}

    g = torch.Generator().manual_seed(0)
    width = lab.layers[0].mlp.down_proj.out_features
    null = {l: max(abs(push(l, torch.randn(width, generator=g).to(lab.dev), s)[0])
                   for _ in range(a.controls) for s in (1, -1))
            for l in sorted({l for l, _ in picks})}

    rows = []
    for layer, n in picks:
        sign = 1 if guess[layer, n] > 0 else -1
        real, lp = push(layer, lab.col(layer, n), sign)
        words = [lab.tok.decode([i]).strip() for i in (lp - base_lp).mean(0).topk(8).indices.tolist()]
        rows.append(dict(layer=layer, n=n, sign=sign, guess=round(abs(guess[layer, n].item()), 3), real=round(real, 3),
                         null=round(null[layer], 3), beats_null=real > null[layer], words=words))
    rows.sort(key=lambda r: -r["real"])
    return dict(model=lab.model_id, pairs=pairs, words=a.words, layers=a.layers, dose=a.dose, base=base, rows=rows)


def main():
    a = parser().parse_args()
    result = search(Lab(a.model, a.bf16), a)
    print(f"{'neuron':<14}{'guess':>7}{'real':>7}{'null':>7}  pushes toward")
    for r in result["rows"]:
        name = f"L{r['layer']} N{r['n']} {'+' if r['sign'] > 0 else '−'}"
        print(f"{'*' if r['beats_null'] else ' '}{name:<13}{r['guess']:>7.2f}{r['real']:>7.2f}{r['null']:>7.2f}  "
              f"{' '.join(r['words'])}")
    path = a.out or f"out/find_{a.model.split('/')[-1]}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(result, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
