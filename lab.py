"""Shared model loading and steering for Qwen3.5 models.

A push adds `dose × median stream norm` along one direction to a layer's MLP output, at every position. For a neuron
the direction is its down_proj column; for a control it is a random unit vector.
"""
import re

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, StoppingCriteria, StoppingCriteriaList

CHAT = [
    "Who are you?",
    "Tell me about your day.",
    "What should I eat for dinner?",
    "Describe the ocean.",
    "Write a short greeting for a friend.",
    "What do you think about computers?",
]
PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Lab:
    def __init__(self, model_id, dtype=torch.float32):
        self.dev = device()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(self.dev).eval()
        self.model.requires_grad_(False)
        self.lm = self.model.model.language_model
        self.layers = self.lm.layers
        self.pushes = {}  # layer -> [hidden] vector added to the MLP output
        self.zeros = None  # layer -> [neurons] zero tensor whose gradient is each neuron's slope
        for i, layer in enumerate(self.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._in_hook(i))
            layer.mlp.down_proj.register_forward_hook(self._out_hook(i))

    def _in_hook(self, i):
        def hook(mod, inp):
            if self.zeros is not None:
                return (inp[0] + self.zeros[i],)
        return hook

    def _out_hook(self, i):
        def hook(mod, inp, out):
            if i in self.pushes:
                return out + self.pushes[i].to(out.dtype)
        return hook

    @property
    def n_layers(self):
        return len(self.layers)

    @property
    def n_neurons(self):
        return self.layers[0].mlp.down_proj.in_features

    def chat_ids(self, messages):
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        return self.tok(text, add_special_tokens=False).input_ids

    def batch(self, seqs):
        """Left-padded ids, mask and positions for a list of token-id lists."""
        T = max(map(len, seqs))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.tensor([[pad] * (T - len(s)) + s for s in seqs], device=self.dev)
        mask = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs], device=self.dev)
        return ids, mask, (mask.cumsum(-1) - 1).clamp(min=0)

    def contexts(self):
        return [self.chat_ids([{"role": "user", "content": c}]) for c in CHAT] + \
               [self.tok(p, add_special_tokens=False).input_ids for p in PLAIN]

    def last_logprobs(self, ids, mask, pos):
        h = self.lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
        return torch.log_softmax(self.model.lm_head(h).float(), -1)

    def stream_norms(self, ids, mask, pos):
        """Median norm of the stream leaving each layer, over real tokens."""
        norms, hooks = {}, []
        for i, layer in enumerate(self.layers):
            hooks.append(layer.register_forward_hook(
                lambda m, inp, out, i=i: norms.__setitem__(i, (out[0] if isinstance(out, tuple) else out)
                                                           [mask.bool()].float().norm(dim=-1).median().item())))
        with torch.inference_mode():
            self.lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
        for h in hooks:
            h.remove()
        return norms

    def col(self, layer, n):
        return self.layers[layer].mlp.down_proj.weight[:, n].float()

    def push_vector(self, layer, direction, dose, norms):
        return direction / direction.norm() * dose * norms[layer]


def parse_push(spec, dose=1.0):
    """'L17 N5244 +0.6, L12 N1339 -' -> [(17, 5244, 0.6), (12, 1339, -dose)]. A bare sign takes `dose`."""
    out = []
    for part in filter(str.strip, spec.split(",")):
        m = re.fullmatch(r"\s*L?(\d+)\D+?N?(\d+)\s*([+\-−])\s*([\d.]*)\s*", part, re.I)
        if not m:
            raise ValueError(f"can't read push {part.strip()!r}; write it like 'L17 N5244 +0.6'")
        size = float(m[4]) if m[4] else dose
        out.append((int(m[1]), int(m[2]), -size if m[3] in "-−" else size))
    return out


def steer_vectors(lab, pushes, norms):
    """[(layer, n, dose)] -> {layer: summed push vector} for lab.pushes."""
    out = {}
    for l, n, d in pushes:
        out[l] = out.get(l, 0) + lab.push_vector(l, lab.col(l, n), d, norms)
    return out


class StopWhen(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


def generate_reply(lab, history, steer, sample=False, tokens=160, streamer=None, stop=None):
    """One assistant turn for `history` with `steer` applied ({} for the normal model). `stop` is a
    threading.Event that ends generation early when set."""
    lab.pushes = steer
    try:
        p = lab.chat_ids(history)
        torch.manual_seed(0)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
        if stop is not None:
            kw["stopping_criteria"] = StoppingCriteriaList([StopWhen(stop)])
        with torch.inference_mode():
            out = lab.model.generate(torch.tensor([p], device=lab.dev), max_new_tokens=tokens,
                                     repetition_penalty=1.1, streamer=streamer, **kw)
    finally:
        lab.pushes = {}
    return lab.tok.decode(out[0, len(p):], skip_special_tokens=True).strip()
