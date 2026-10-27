import contextlib
import re

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, StoppingCriteria, StoppingCriteriaList

from config import PROMPTS, TEXTS


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Lab:
    def __init__(self, model_id, bf16=False):
        self.model_id = model_id
        self.dev = device()
        dtype = torch.bfloat16 if bf16 or self.dev == "cuda" or model_id.endswith("4B") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(self.dev).eval()
        self.model.requires_grad_(False)
        self.lm = self.model.model.language_model
        self.layers = self.lm.layers
        self.n_layers = len(self.layers)
        self.n_neurons = self.layers[0].mlp.down_proj.in_features
        self.pushes = {}
        self.zeros = None
        for i, layer in enumerate(self.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._grad_hook(i))
            layer.mlp.down_proj.register_forward_hook(self._push_hook(i))

    def _grad_hook(self, i):
        def hook(mod, inp):
            if self.zeros is not None:
                return (inp[0] + self.zeros[i],)
        return hook

    def _push_hook(self, i):
        def hook(mod, inp, out):
            if i in self.pushes:
                return out + self.pushes[i].to(out.dtype)
        return hook

    def chat_ids(self, messages):
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        return self.tok(text, add_special_tokens=False).input_ids

    def text_ids(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def prompts(self):
        return [self.chat_ids([{"role": "user", "content": p}]) for p in PROMPTS]

    def batch(self, seqs):
        T = max(map(len, seqs))
        pad = self.tok.pad_token_id or 0
        ids = torch.tensor([[pad] * (T - len(s)) + s for s in seqs], device=self.dev)
        mask = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs], device=self.dev)
        return ids, mask, (mask.cumsum(-1) - 1).clamp(min=0)

    def norm_batch(self):
        return self.batch(self.prompts() + [self.text_ids(t) for t in TEXTS])

    def last_logprobs(self, ids, mask, pos):
        h = self.lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
        return torch.log_softmax(self.model.lm_head(h).float(), -1)

    def layer_outputs(self, ids, mask, pos):
        outs = {}
        hooks = [layer.register_forward_hook(lambda m, inp, out, i=i: outs.__setitem__(i, hidden(out).detach()))
                 for i, layer in enumerate(self.layers)]
        try:
            with torch.inference_mode():
                self.lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return outs

    def stream_norms(self, ids, mask, pos):
        real = mask.bool()
        return {i: h[real].float().norm(dim=-1).median().item() for i, h in self.layer_outputs(ids, mask, pos).items()}

    @contextlib.contextmanager
    def start_after(self, layer, stream):
        for i in range(layer):
            self.layers[i].forward = lambda hidden_states, *a, **k: hidden_states
        self.layers[layer].forward = lambda hidden_states, *a, **k: stream.to(hidden_states.dtype)
        try:
            yield
        finally:
            for i in range(layer + 1):
                del self.layers[i].forward

    def col(self, layer, n):
        return self.layers[layer].mlp.down_proj.weight[:, n].float()

    def push_vector(self, layer, direction, dose, norms):
        return direction / direction.norm() * dose * norms[layer]


def hidden(out):
    return out[0] if isinstance(out, tuple) else out


def parse_push(spec, dose=1.0):
    out = []
    for part in filter(str.strip, spec.split(",")):
        m = re.fullmatch(r"\s*L?(\d+)\D+?N?(\d+)\s*([+\-−])\s*([\d.]*)\s*", part, re.I)
        if not m:
            raise ValueError(f"can't read push {part.strip()!r}; write it like 'L17 N5244 +0.6'")
        size = float(m[4]) if m[4] else dose
        out.append((int(m[1]), int(m[2]), -size if m[3] in "-−" else size))
    return out


def steer_vectors(lab, pushes, norms):
    out = {}
    for l, n, d in pushes:
        out[l] = out.get(l, 0) + lab.push_vector(l, lab.col(l, n), d, norms)
    return out


def layer_range(spec, n_layers):
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", spec or f"0-{n_layers - 1}")
    lo, hi = (int(m[1]), int(m[2] or m[1])) if m else (-1, -1)
    if not 0 <= lo <= hi < n_layers:
        raise ValueError(f"layers {spec!r} must look like 3-23 within 0-{n_layers - 1}")
    return lo, hi


class StopWhen(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


def generate_reply(lab, history, steer, sample=False, tokens=160, streamer=None, stop=None):
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
