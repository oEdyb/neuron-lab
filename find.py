"""Find neurons that push a set of words, then check the best ones with real pushes.

    uv run python find.py                                  # pirate words on Qwen3.5-0.8B
    uv run python find.py --words "alien,aliens,planet,galaxy" --demo
    uv run python find.py --text "Arr, ahoy matey! I be a pirate captain, sailin' the seven seas for treasure." --layers 0-15

1. Gradient: one backward pass scores every neuron in every layer by how much a push would raise the target words'
   probability (a straight-line guess).
2. Check: the top candidates get a real ±dose push, next to random directions in the same layers.
3. Demo (optional): the best checked neuron answers "Who are you?" at a few doses.
"""
import argparse
import json
import os
import re

import torch

from lab import CHAT, Lab

PIRATE = "Arr,Ahoy,Aye,Avast,Yarr, Arr, ahoy, Ahoy, matey, Matey, ye, aye, Aye, avast, yarr, hearties, landlubber, plunder, buccaneer"


def target_ids(tok, words):
    keep, drop = [], []
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        (keep if len(ids) == 1 else drop).append((w, ids))
    return sorted({ids[0] for _, ids in keep}), [w for w, _ in keep], [w for w, _ in drop]


def target_mass(lp, targets):
    """Log of the total probability on the target tokens, averaged over contexts."""
    return torch.logsumexp(lp[:, targets], -1).mean()


NEUTRAL = "Hello! I'm an AI assistant made to help with questions and tasks. How can I help you today?"


def reply_logprob(lab, prompts, reply):
    """Mean log-prob per token of `reply` as the answer to each prompt (teacher forced)."""
    r = lab.tok(reply, add_special_tokens=False).input_ids
    seqs = [p + r for p in prompts]
    ids, mask, pos = lab.batch(seqs)
    h = lab.lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state
    lp = torch.log_softmax(lab.model.lm_head(h[:, -len(r) - 1:-1]).float(), -1)
    return lp.gather(-1, ids[:, -len(r):, None]).mean()


def layer_range(spec, n_layers):
    """'3-23' or '14' -> (lo, hi), inclusive; ValueError with the valid range otherwise."""
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", spec)
    lo, hi = (int(m[1]), int(m[2] or m[1])) if m else (-1, -1)
    if not (0 <= lo <= hi < n_layers):
        raise ValueError(f"layers {spec!r} must look like 3-23 within 0-{n_layers - 1}")
    return lo, hi


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--bf16", action="store_true", help="half the memory; for 4B on a 24 GB Mac")
    ap.add_argument("--words", default=PIRATE, help="comma-separated; a leading space matters")
    ap.add_argument("--dose", type=float, default=0.6, help="push size as a fraction of the median stream norm")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--controls", type=int, default=4, help="random directions per candidate layer")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--text", help="score whole replies instead of next words; several separated by |")
    ap.add_argument("--neutral", help="matched neutral replies, same count and shape as --text, separated by |")
    ap.add_argument("--pairs", help="file with one 'target|neutral' pair per line; fills --text and --neutral")
    ap.add_argument("--layers", help="only search these layers, e.g. 0-15")
    ap.add_argument("--out", help="where to write the JSON result (default: out/find_<model>_<tag>.json)")
    return ap


def search(lab, a, log=print):
    """Run the search on a loaded Lab with parsed args; returns the result dict that find.py saves."""
    if a.pairs:
        rows = [l.rstrip("\n").split("|") for l in open(a.pairs) if "|" in l]
        a.text, a.neutral = "|".join(r[0] for r in rows), "|".join(r[1] for r in rows)
    targets, kept, dropped = target_ids(lab.tok, a.words.split(","))
    log(f"{a.model} on {lab.dev}: {lab.n_layers} layers × {lab.n_neurons} neurons")
    if not a.text:
        log(f"target tokens: {kept}" + (f"; dropped (more than one token): {dropped}" if dropped else ""))

    ids, mask, pos = lab.batch(lab.contexts())
    norms = lab.stream_norms(ids, mask, pos)
    chat = lab.contexts()[:len(CHAT)]

    def objective():
        if a.text:  # how much more the model favours each target reply over its matched neutral one
            pairs = list(zip(a.text.split("|"), (a.neutral or NEUTRAL).split("|"))) if a.neutral \
                else [(t, NEUTRAL) for t in a.text.split("|")]
            return sum(reply_logprob(lab, chat, t) - reply_logprob(lab, chat, n) for t, n in pairs) / len(pairs)
        return target_mass(lab.last_logprobs(ids, mask, pos), targets)

    with torch.inference_mode():
        base = objective().item()

    # 1. gradient: slope of the target mass for every neuron, as a push of `dose` would apply it
    lab.zeros = [torch.zeros(lab.n_neurons, device=lab.dev, requires_grad=True) for _ in range(lab.n_layers)]
    try:
        objective().backward()
        guess = torch.stack([
            z.grad * a.dose * norms[l] / lab.layers[l].mlp.down_proj.weight.float().norm(dim=0)
            for l, z in enumerate(lab.zeros)])  # [layers, neurons]: predicted change for a +dose push
    finally:
        lab.zeros = None
    if a.layers:
        lo, hi = layer_range(a.layers, lab.n_layers)
        allowed = torch.zeros(lab.n_layers, 1, device=lab.dev)
        allowed[lo:hi + 1] = 1
        guess = guess * allowed
    flat = guess.abs().flatten().topk(a.top).indices
    cands = [(int(i) // lab.n_neurons, int(i) % lab.n_neurons) for i in flat]

    # 2. check: real pushes on the candidates, and random directions in the same layers
    def real(layer, direction, sign):
        lab.pushes = {layer: lab.push_vector(layer, direction, sign * a.dose, norms)}
        try:
            with torch.inference_mode():
                lp = lab.last_logprobs(ids, mask, pos)
                score = objective().item()
        finally:
            lab.pushes = {}
        return score - base, lp

    g = torch.Generator(device="cpu").manual_seed(0)
    hidden = lab.layers[0].mlp.down_proj.out_features
    null = {}
    for layer in sorted({l for l, _ in cands}):
        null[layer] = max(abs(real(layer, torch.randn(hidden, generator=g).to(lab.dev), s)[0])
                          for _ in range(a.controls) for s in (1, -1))

    base_lp = None
    with torch.inference_mode():
        base_lp = lab.last_logprobs(ids, mask, pos)
    rows = []
    for layer, n in cands:
        sign = 1 if guess[layer, n] > 0 else -1
        delta, lp = real(layer, lab.col(layer, n), sign)
        words = [lab.tok.decode([i]).strip() for i in (lp - base_lp).mean(0).topk(8).indices.tolist()]
        rows.append(dict(layer=layer, n=n, sign=sign, guess=round(abs(guess[layer, n].item()), 3),
                         real=round(delta, 3), null=round(null[layer], 3), words=words))
    rows.sort(key=lambda r: -r["real"])
    for r in rows:
        r["beats_null"] = r["real"] > r["null"]
    return dict(model=a.model, words=None if a.text else kept, text=a.text, neutral=a.neutral, layers=a.layers,
                dose=a.dose, base=base, rows=rows, norms=norms)


def main():
    a = parser().parse_args()
    lab = Lab(a.model, torch.bfloat16 if a.bf16 else torch.float32)
    result = search(lab, a)
    rows, base, norms = result["rows"], result["base"], result.pop("norms")
    print(f"\nbaseline score: {base:.2f}. dose {a.dose}. real > null means it beats random directions.\n")
    print(f"{'neuron':<14}{'guess':>7}{'real':>7}{'null':>7}  top words when pushed")
    for r in rows:
        mark = "*" if r["beats_null"] else " "
        name = f"L{r['layer']} N{r['n']} {'+' if r['sign'] > 0 else '−'}"
        print(f"{mark}{name:<13}{r['guess']:>7.2f}{r['real']:>7.2f}{r['null']:>7.2f}  {' '.join(r['words'])}")

    tag = "text" if a.text else result["words"][0].strip().lower()
    path = a.out or f"out/find_{a.model.split('/')[-1]}_{tag}{'_L' + a.layers if a.layers else ''}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(result, open(path, "w"), indent=1)
    print(f"\nsaved {path}")

    if a.demo:
        best = rows[0]
        print(f"\ndemo: L{best['layer']} N{best['n']}, sampled, temperature 0.7")
        prompt = lab.chat_ids([{"role": "user", "content": "Who are you?"}])
        for dose in (0.0, 0.4, 0.6, 0.8):
            if dose:
                lab.pushes = {best["layer"]: lab.push_vector(best["layer"], lab.col(best["layer"], best["n"]),
                                                             best["sign"] * dose, norms)}
            torch.manual_seed(0)
            with torch.inference_mode():
                out = lab.model.generate(torch.tensor([prompt], device=lab.dev), max_new_tokens=80, do_sample=True,
                                         temperature=0.7, top_p=0.9)
            lab.pushes = {}
            text = lab.tok.decode(out[0, len(prompt):], skip_special_tokens=True).strip().replace("\n", " ")
            print(f"\n[{best['sign'] * dose:+.1f}] {text}")


if __name__ == "__main__":
    main()
