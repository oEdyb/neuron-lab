"""Multi-turn chat with pushed neurons, steered next to normal.

    uv run python chat.py --push "L20 N365 -,L17 N338 +" --dose 0.3                # type your turns
    uv run python chat.py --push "..." --dose 0.3 --say "Hi!|Who are you?|Why?"     # scripted turns

A push is "L17 N5244 +0.6"; a bare sign ("L17 N5244 +") takes --dose. Both sides keep their own history; your turns are shared.
"""
import argparse
import json

import torch

from lab import Lab, generate_reply, parse_push, steer_vectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--push", required=True)
    ap.add_argument("--dose", type=float, default=0.3)
    ap.add_argument("--say", help="scripted user turns separated by |")
    ap.add_argument("--say-json", help='scripted user turns as a JSON list, e.g. \'["Hi", "Why?"]\'')
    ap.add_argument("--sample", action="store_true", help="sample at temperature 0.7 instead of greedy")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--no-normal", action="store_true", help="only show the steered side")
    a = ap.parse_args()

    lab = Lab(a.model, torch.bfloat16 if a.bf16 else torch.float32)
    ids, mask, pos = lab.batch(lab.contexts())
    norms = lab.stream_norms(ids, mask, pos)
    steer = steer_vectors(lab, parse_push(a.push, a.dose), norms)

    def reply(history, pushed):
        return generate_reply(lab, history, steer if pushed else {}, a.sample, a.tokens)

    sides = [("steered", True)] + ([] if a.no_normal else [("normal", False)])
    hist = {name: [] for name, _ in sides}
    turns = iter(json.loads(a.say_json)) if a.say_json else iter(a.say.split("|")) if a.say else None
    print(f"{a.model} · {a.push} × {a.dose} · {'sampled' if a.sample else 'greedy'}\n")
    while True:
        try:
            user = next(turns) if turns else input("you> ")
        except (StopIteration, EOFError, KeyboardInterrupt):
            break
        if turns:
            print(f"you> {user}")
        for name, pushed in sides:
            hist[name].append({"role": "user", "content": user})
            text = reply(hist[name], pushed)
            hist[name].append({"role": "assistant", "content": text})
            print(f"[{name}] {text}\n", flush=True)


if __name__ == "__main__":
    main()
