import argparse
import json

from config import DOSE
from lab import Lab, generate_reply, parse_push, steer_vectors


def main():
    ap = argparse.ArgumentParser(description="Chat with pushed neurons, next to the normal model.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--push", required=True, help="e.g. 'L17 N5244 +0.6, L12 N1339 -'")
    ap.add_argument("--dose", type=float, default=DOSE, help="dose for pushes written with a bare sign")
    ap.add_argument("--say", help="scripted turns as a JSON list instead of typing")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--no-normal", action="store_true")
    a = ap.parse_args()

    lab = Lab(a.model, a.bf16)
    steer = steer_vectors(lab, parse_push(a.push, a.dose), lab.stream_norms(*lab.norm_batch()))
    sides = {"steered": steer} if a.no_normal else {"steered": steer, "normal": {}}
    history = {side: [] for side in sides}
    turns = iter(json.loads(a.say)) if a.say else None
    while True:
        try:
            user = next(turns) if turns else input("you> ")
        except (StopIteration, EOFError, KeyboardInterrupt):
            break
        if turns:
            print(f"you> {user}")
        for side, vec in sides.items():
            history[side].append({"role": "user", "content": user})
            text = generate_reply(lab, history[side], vec, a.sample, a.tokens)
            history[side].append({"role": "assistant", "content": text})
            print(f"[{side}] {text}\n", flush=True)


if __name__ == "__main__":
    main()
