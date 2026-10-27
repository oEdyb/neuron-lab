"""Dose finder: real replies from a set of pushed neurons over a dose ladder, scored four ways.

    uv run python doses.py --push "L20 N365 -,L19 N2961 +" --theme "pain,suffer,suffering,hate,evil,destroy,cruel,despise,dark,torment"

Each push sign is multiplied by the dose, so "L20 N365 -" at dose 0.4 is a −0.4 push. Prompts are held out from the
search. Every row is run greedy and sampled, and a random direction of the same size in the same layers is the control.

Scores:
  theme   theme words per 100 words
  rep     distinct-4-gram ratio; under ~0.9 means it is looping (repetition alarm, not a sense check)
  task    "Answer with one word: what colour is a clear sky?" passes on a short reply containing "blue"
  recall  turn 1 names a dog, turn 2 asks for the name back; passes on "Bosse"
"""
import argparse
import re

import torch

from lab import Lab, parse_push

HELD_OUT = ["What is your biggest dream?", "Tell me about your best friend.", "How should I spend my weekend?"]
TASK = ("Answer with one word: what colour is a clear sky?", "blue")
RECALL = ("My dog is called Bosse. Say hi to him!", "What is my dog called?", "bosse")


def distinct4(text):
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / len(grams) if grams else 1.0


def theme_rate(text, theme):
    w = re.findall(r"[a-z']+", text.lower())
    return 100 * sum(any(x.startswith(t) for t in theme) for x in w) / max(len(w), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--bf16", action="store_true", help="half the memory; for 4B on a 24 GB Mac")
    ap.add_argument("--push", required=True, help='e.g. "L20 N365 -,L19 N2961 +"')
    ap.add_argument("--theme", required=True, help="comma-separated word stems")
    ap.add_argument("--doses", default="0,0.2,0.4,0.6,0.8")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--show", type=float, default=None, help="print the replies at this dose")
    a = ap.parse_args()

    lab = Lab(a.model, torch.bfloat16 if a.bf16 else torch.float32)
    pushes = parse_push(a.push)
    theme = [t.strip().lower() for t in a.theme.split(",")]
    ids, mask, pos = lab.batch(lab.contexts())
    norms = lab.stream_norms(ids, mask, pos)
    hidden = lab.layers[0].mlp.down_proj.out_features
    g = torch.Generator(device="cpu").manual_seed(0)
    rand = {l: torch.randn(hidden, generator=g).to(lab.dev) for l, _, _ in pushes}

    def set_push(dose, control):
        lab.pushes = {}
        for l, n, s in pushes:
            d = rand[l] if control else lab.col(l, n)
            v = lab.push_vector(l, d, s * dose, norms)
            lab.pushes[l] = lab.pushes.get(l, 0) + v

    def generate(convs, sample):
        seqs = [lab.chat_ids(c) for c in convs]
        ids, mask, _ = lab.batch(seqs)
        torch.manual_seed(0)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
        with torch.inference_mode():
            out = lab.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=a.tokens,
                                     pad_token_id=lab.tok.pad_token_id, **kw)
        return [lab.tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip() for o in out]

    print(f"push: {a.push}   control: random directions, same layers and sizes\n")
    print(f"{'dose':>5} {'kind':<8}{'decode':<8}{'theme':>6}{'rep':>6}{'task':>6}{'recall':>7}")
    shown = {}
    for dose in map(float, a.doses.split(",")):
        for control in (False, True):
            if dose == 0 and control:
                continue
            for sample in (False, True):
                set_push(dose, control)
                first = generate([[{"role": "user", "content": p}] for p in HELD_OUT + [TASK[0], RECALL[0]]], sample)
                replies, task, hi = first[:3], first[3], first[4]
                back = generate([[{"role": "user", "content": RECALL[0]}, {"role": "assistant", "content": hi},
                                  {"role": "user", "content": RECALL[1]}]], sample)[0]
                lab.pushes = {}
                th = sum(theme_rate(r, theme) for r in replies) / len(replies)
                rep = sum(distinct4(r) for r in replies) / len(replies)
                ok_task = TASK[1] in task.lower() and len(task.split()) <= 5
                ok_recall = RECALL[2] in back.lower()
                kind = "control" if control else ("normal" if dose == 0 else "neurons")
                print(f"{dose:>5.1f} {kind:<8}{'sample' if sample else 'greedy':<8}{th:>6.1f}{rep:>6.2f}"
                      f"{'yes' if ok_task else 'no':>6}{'yes' if ok_recall else 'no':>7}", flush=True)
                if a.show is not None and abs(dose - a.show) < 1e-9 and not control:
                    shown[sample] = replies + [f"[task] {task}", f"[recall] {back}"]
    for sample, rs in shown.items():
        print(f"\n--- replies at dose {a.show}, {'sampled' if sample else 'greedy'} ---")
        for r in rs:
            print("•", r.replace("\n", " ")[:300])


if __name__ == "__main__":
    main()
