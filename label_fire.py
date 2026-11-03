"""Does a neuron fire on text its label predicts? Goodfire's held-out test, without a corpus.

A writer sees only the label and writes five examples and five near misses. The neuron is read
before any push, so this checks what makes it fire. Steering scores check what pushing it does.
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import torch

from claude_cli import ClaudeFailed, ask
from config import PROMPTS, TEXTS
from lab import Lab, parse_push

COUNT = 5
WRITER_MODEL = "claude-haiku-5-5"
PASS_AUC = 0.8
PASS_P = 0.05
PASS_CONTROL_SHARE = 0.9
PROMPT = """A neuron in a language model has this label: {label!r}

Write text to test the label before anyone has measured the neuron.
- "examples": {count} varied sentences the label predicts should drive the neuron.
- "near_misses": {count} sentences that share topic, wording or length with the examples but lack \
what the label describes.
Each sentence is 8 to 25 words. Don't reuse a sentence or quote the label."""
SCHEMA = {
    "type": "object",
    "properties": {side: {"type": "array", "items": {"type": "string"}, "minItems": COUNT, "maxItems": COUNT}
                   for side in ("examples", "near_misses")},
    "required": ["examples", "near_misses"],
    "additionalProperties": False,
}


class LabelFireError(Exception):
    def __init__(self, code, error, fix):
        super().__init__(error)
        self.code, self.error, self.fix = code, error, fix


def claude_writer(model=WRITER_MODEL):
    """The writer sees only the label."""
    def write(label):
        try:
            return ask(PROMPT.format(label=label, count=COUNT), SCHEMA, model)
        except ClaudeFailed as e:
            raise LabelFireError("writer_failed", str(e),
                                 "Check `claude -p hi` works, or pass --texts with your own sentences") from e
    return write


def file_writer(path):
    return lambda label: json.loads(Path(path).read_text())


def check_texts(raw):
    fix = f'Give {{"examples":[{COUNT} strings],"near_misses":[{COUNT} strings]}}, all different'
    if not isinstance(raw, dict) or set(raw) != {"examples", "near_misses"}:
        raise LabelFireError("bad_texts", "the writer must return examples and near_misses", fix)
    texts = raw["examples"] + raw["near_misses"]
    if (len(raw["examples"]), len(raw["near_misses"])) != (COUNT, COUNT):
        raise LabelFireError("bad_texts", f"need exactly {COUNT} examples and {COUNT} near misses", fix)
    if not all(isinstance(t, str) and t.strip() for t in texts) or len(set(texts)) != len(texts):
        raise LabelFireError("bad_texts", "every text must be a different, non-empty string", fix)
    return raw


def capture(lab, layer, texts, chunk=8):
    """One tensor per text, tokens by neurons: the values each neuron feeds into down_proj."""
    saved = []
    hook = lab.layers[layer].mlp.down_proj.register_forward_pre_hook(
        lambda module, inputs: saved.append(inputs[0].detach().float().cpu()))
    lab.pushes = {}
    out = []
    try:
        for start in range(0, len(texts), chunk):
            seqs = [lab.text_ids(t) for t in texts[start:start + chunk]]
            ids, mask, pos = lab.batch(seqs)
            with torch.inference_mode():
                lab.lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
            acts = saved.pop()
            out += [(row[real.bool().cpu()], seq) for row, real, seq in zip(acts, mask, seqs)]
    finally:
        hook.remove()
    return out


def peak(captured, neuron, sign):
    """Strongest value on the neuron's pole. The first token has no context, so it doesn't count."""
    acts, seq = captured
    if len(seq) < 2:
        raise LabelFireError("bad_texts", "each text needs at least two tokens", "Write full sentences")
    values = sign * acts[1:, neuron]
    top = values.argmax().item()
    return values[top].item(), seq[top + 1]


def auc(pos, neg):
    """Chance an example outranks a near miss. 1 separates them fully, 0.5 is no better than a coin."""
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def permutation_p(pos, neg):
    """Exact chance a random split of the same values does at least this well."""
    values = pos + neg
    observed = auc(pos, neg)
    splits = list(itertools.combinations(range(len(values)), len(pos)))
    hits = 0
    for chosen in splits:
        picked = set(chosen)
        a = [v for i, v in enumerate(values) if i in picked]
        b = [v for i, v in enumerate(values) if i not in picked]
        hits += auc(a, b) >= observed - 1e-12
    return hits / len(splits)


def verdict(score, p, control_share):
    if score <= 0.5:
        return "fail"
    return "pass" if score >= PASS_AUC and p <= PASS_P and control_share >= PASS_CONTROL_SHARE else "weak"


def run(lab, label, layer, n, sign, writer, controls=50, seed=0):
    if not 0 <= layer < lab.n_layers or not 0 <= n < lab.neuron_count(layer):
        raise LabelFireError("bad_neuron", f"L{layer} N{n} isn't in {lab.model_id}", "Pick a neuron from /api/top")
    texts = check_texts(writer(label))
    reference = PROMPTS + TEXTS
    written = texts["examples"] + texts["near_misses"]
    captured = capture(lab, layer, written + reference)
    ours, base = captured[:len(written)], captured[len(written):]
    threshold = max(peak(c, n, sign)[0] for c in base)

    def row(text, c):
        value, token = peak(c, n, sign)
        return {"text": text, "peak": round(value, 4), "top_token": lab.tok.decode([token]),
                "fired": value > max(threshold, 0)}

    rows = [row(t, c) for t, c in zip(written, ours)]
    examples, near = rows[:COUNT], rows[COUNT:]
    pos, neg = [peak(c, n, sign)[0] for c in ours[:COUNT]], [peak(c, n, sign)[0] for c in ours[COUNT:]]
    score = auc(pos, neg)

    others = [i for i in range(lab.neuron_count(layer)) if i != n]
    picked = random.Random(seed).sample(others, min(controls, len(others)))
    control = [auc([peak(c, i, sign)[0] for c in ours[:COUNT]], [peak(c, i, sign)[0] for c in ours[COUNT:]])
               for i in picked]
    share = sum(c < score for c in control) / len(control) if control else 0.0
    p = permutation_p(pos, neg)
    return {
        "ok": True, "model": lab.model_id, "layer": layer, "n": n, "sign": "+" if sign > 0 else "-", "label": label,
        "verdict": verdict(score, p, share),
        "auc": round(score, 3), "p": round(p, 4),
        "example_fire_rate": sum(r["fired"] for r in examples) / COUNT,
        "near_miss_fire_rate": sum(r["fired"] for r in near) / COUNT,
        "peak_gap": round(sum(pos) / COUNT - sum(neg) / COUNT, 4),
        "fire_threshold": round(max(threshold, 0), 4),
        "random_neurons": {"count": len(control), "mean_auc": round(sum(control) / len(control), 3) if control else None,
                           "share_beaten": round(share, 3)},
        "examples": examples, "near_misses": near,
        "means": "what makes the neuron fire at this layer, not what pushing it does",
    }


def main():
    ap = argparse.ArgumentParser(description="Score whether one neuron fires on what its label predicts.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--push", required=True, help="one neuron and pole, e.g. 'L17 N5244 +'")
    ap.add_argument("--label", required=True, help="what the neuron is claimed to do; the writer sees only this")
    ap.add_argument("--texts", help="JSON with examples and near_misses, instead of asking the writer")
    ap.add_argument("--writer-model", default=WRITER_MODEL)
    ap.add_argument("--controls", type=int, default=50, help="random neurons in the same layer to score on the same texts")
    a = ap.parse_args()
    try:
        pushes = parse_push(a.push)
        if len(pushes) != 1:
            raise LabelFireError("bad_push", "this test reads one neuron at a time", "Pass one neuron in --push")
        layer, n, size = pushes[0]
        writer = file_writer(a.texts) if a.texts else claude_writer(a.writer_model)
        result = run(Lab(a.model, a.bf16), a.label, layer, n, 1 if size > 0 else -1, writer, a.controls)
    except LabelFireError as e:
        print(json.dumps({"ok": False, "code": e.code, "error": e.error, "fix": e.fix}))
        sys.exit(1)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
