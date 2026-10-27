import argparse
import re

import torch

from config import HELD_OUT, RECALL, TASK
from lab import Lab, parse_push, steer_vectors


def distinct4(text):
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / len(grams) if grams else 1.0


def theme_rate(text, theme):
    w = re.findall(r"[a-z']+", text.lower())
    return 100 * sum(any(x.startswith(t) for t in theme) for x in w) / max(len(w), 1)


def main():
    ap = argparse.ArgumentParser(description="Try pushes over a dose ladder and score the replies.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--push", required=True, help="neurons and signs, e.g. 'L17 N5244 +, L12 N1339 -'")
    ap.add_argument("--theme", required=True, help="comma-separated word stems to count, e.g. 'evil,destroy,hate'")
    ap.add_argument("--doses", default="0,0.2,0.4,0.6,0.8")
    ap.add_argument("--tokens", type=int, default=100)
    a = ap.parse_args()

    lab = Lab(a.model, a.bf16)
    pushes = parse_push(a.push)
    theme = [t.strip().lower() for t in a.theme.split(",")]
    norms = lab.stream_norms(*lab.norm_batch())
    width = lab.layers[0].mlp.down_proj.out_features
    g = torch.Generator().manual_seed(0)
    rand = {l: torch.randn(width, generator=g).to(lab.dev) for l, _, _ in pushes}

    def generate(convs, sample):
        ids, mask, _ = lab.batch([lab.chat_ids(c) for c in convs])
        torch.manual_seed(0)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
        with torch.inference_mode():
            out = lab.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=a.tokens,
                                     pad_token_id=lab.tok.pad_token_id, **kw)
        return [lab.tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip() for o in out]

    print(f"{'dose':>5} {'kind':<8}{'decode':<8}{'theme':>6}{'rep':>6}{'task':>6}{'recall':>7}")
    for dose in map(float, a.doses.split(",")):
        for control in ([False] if dose == 0 else [False, True]):
            for sample in (False, True):
                if control:
                    lab.pushes = {}
                    for l, _, s in pushes:
                        lab.pushes[l] = lab.pushes.get(l, 0) + lab.push_vector(l, rand[l], s * dose, norms)
                else:
                    lab.pushes = steer_vectors(lab, [(l, n, s * dose) for l, n, s in pushes], norms)
                try:
                    first = generate([[{"role": "user", "content": p}] for p in HELD_OUT + [TASK[0], RECALL[0]]], sample)
                    back = generate([[{"role": "user", "content": RECALL[0]}, {"role": "assistant", "content": first[-1]},
                                      {"role": "user", "content": RECALL[1]}]], sample)[0]
                finally:
                    lab.pushes = {}
                replies, task = first[:len(HELD_OUT)], first[len(HELD_OUT)]
                kind = "control" if control else "normal" if dose == 0 else "neurons"
                print(f"{dose:>5.1f} {kind:<8}{'sample' if sample else 'greedy':<8}"
                      f"{sum(theme_rate(r, theme) for r in replies) / len(replies):>6.1f}"
                      f"{sum(distinct4(r) for r in replies) / len(replies):>6.2f}"
                      f"{'yes' if TASK[1] in task.lower() and len(task.split()) <= 5 else 'no':>6}"
                      f"{'yes' if RECALL[2] in back.lower() else 'no':>7}", flush=True)


if __name__ == "__main__":
    main()
