import contextlib
import re

import torch
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
from transformers import (TextStreamer, AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration,
                          Qwen3_5ForConditionalGeneration, StoppingCriteria, StoppingCriteriaList)

from config import LOOP_REPEATS, LOOP_MAX_NGRAM, PROMPTS, TEXTS


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def map_delta(query, key, value, g, beta, reference, chunk_size=32, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    if query.shape[1] > chunk_size or initial_state is not None or output_final_state:
        return reference(query, key, value, g=g, beta=beta, initial_state=initial_state,
                         output_final_state=output_final_state,
                         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
    dtype, length = query.dtype, query.shape[1]
    q, k, v, b, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                          for x in (query, key, value, beta, g)]
    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q, dim=-1, eps=1e-6), l2norm(k, dim=-1, eps=1e-6)
    q = q * q.shape[-1] ** -0.5
    padding = chunk_size - length
    q, k, v = [F.pad(x, (0, 0, 0, padding)) for x in (q, k, v)]
    b, decay = F.pad(b, (0, padding)), F.pad(decay, (0, padding))
    kb, vb = (k * b[..., None]).unsqueeze(2), (v * b[..., None]).unsqueeze(2)
    q, k = q.unsqueeze(2), k.unsqueeze(2)
    cumulative = decay.unsqueeze(2).cumsum(-1)
    mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
    weights = (cumulative[..., None] - cumulative[..., None, :]).masked_fill(mask, float("-inf")).exp()
    system = (kb @ k.transpose(-1, -2)) * weights
    attention = (q @ k.transpose(-1, -2)) * weights
    values = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
    out = (attention @ values).squeeze(2)[:, :, :length].transpose(1, 2)
    return out.to(dtype, memory_format=torch.contiguous_format), None


class Lab:
    def __init__(self, model_id, bf16=False):
        self.model_id = model_id
        self.dev = device()
        dtype = torch.bfloat16 if bf16 or self.dev == "cuda" or model_id.endswith("4B") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        self.is_gemma = config.model_type == "gemma4"
        if self.is_gemma:
            dtype = torch.bfloat16 if bf16 else torch.float32
            config.vision_config = config.audio_config = None
            self.model = Gemma4ForConditionalGeneration.from_pretrained(
                model_id, config=config, dtype=dtype).eval()
            for name, module in self.model.model.language_model.named_children():
                if name != "embed_tokens_per_layer" or self.dev != "cuda":
                    module.to(self.dev)
            self.model.lm_head.to(self.dev)
            if self.dev == "cuda":
                ple = self.model.model.language_model.embed_tokens_per_layer
                ple.register_forward_pre_hook(lambda m, inp: (inp[0].cpu(),))
                ple.register_forward_hook(lambda m, inp, out: out.to(self.dev))
        else:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(self.dev).eval()
        self.model.requires_grad_(False)
        self.lm = self.model.model.language_model
        self.layers = self.lm.layers
        self.n_layers = len(self.layers)
        self.neurons_per_layer = [layer.mlp.down_proj.in_features for layer in self.layers]
        self.n_neurons = (self.neurons_per_layer[0] if len(set(self.neurons_per_layer)) == 1
                          else self.neurons_per_layer)
        self.pushes = {}
        self.zeros = None
        for i, layer in enumerate(self.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._grad_hook(i))
            layer.mlp.down_proj.register_forward_hook(self._push_hook(i))

    def _grad_hook(self, i):
        def hook(mod, inp):
            if self.zeros is not None:
                return (inp[0] + self.zeros[i].to(inp[0].dtype),)
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

    def neuron_count(self, layer):
        return self.neurons_per_layer[layer]

    def logits(self, h):
        logits = self.model.lm_head(h)
        if self.is_gemma:
            cap = self.lm.config.final_logit_softcapping
            if cap is not None:
                logits = (logits / cap).tanh() * cap
        return logits

    def last_logprobs(self, ids, mask, pos):
        h = self.lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
        return torch.log_softmax(self.logits(h).float(), -1)

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
        if self.is_gemma:
            raise ValueError("Gemma mapping requires full forwards for per-layer inputs and shared KV")
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


def repeating(ids):
    return any(len(ids) >= n * LOOP_REPEATS and
               ids[-n:] * LOOP_REPEATS == ids[-n * LOOP_REPEATS:]
               for n in range(1, LOOP_MAX_NGRAM + 1))


class RowStreamer(TextStreamer):
    def __init__(self, tok, emit):
        super().__init__(tok, skip_special_tokens=True)
        self.emit = emit

    def on_finalized_text(self, text, stream_end=False):
        if text:
            self.emit(text)


class BatchStream(StoppingCriteria):
    def __init__(self, lab, columns, tokens, emit, stop):
        self.lab, self.columns, self.tokens = lab, columns, tokens
        self.emit, self.stop = emit, stop
        self.ids = [[] for _ in columns]
        self.reasons = [None for _ in columns]
        self.rows = [RowStreamer(lab.tok, lambda text, c=c: emit(dict(col=c["id"], delta=text))) for c in columns]
        eos = lab.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.prompt = True

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        for i, token in enumerate(value.tolist()):
            if self.reasons[i] is not None:
                continue
            self.ids[i].append(token)
            self.rows[i].put(torch.tensor([token]))
            reason = ("client" if self.stop is not None and self.stop.is_set() else
                      "eos" if token in self.eos else "loop" if repeating(self.ids[i]) else
                      "length" if len(self.ids[i]) >= self.tokens else None)
            if reason:
                self.finish(i, reason)

    def finish(self, i, reason):
        self.reasons[i] = reason
        self.rows[i].end()
        self.emit(dict(col=self.columns[i]["id"], done=True,
                       text=self.lab.tok.decode(self.ids[i], skip_special_tokens=True).strip(), stopped=reason))

    def end(self):
        for i, reason in enumerate(self.reasons):
            if reason is None:
                self.finish(i, "client" if self.stop is not None and self.stop.is_set() else "length")

    def __call__(self, input_ids, scores, **kwargs):
        return torch.tensor([r is not None for r in self.reasons], device=input_ids.device)


def generate_compare(lab, columns, vectors, sample=False, tokens=300, emit=None, stop=None):
    events = []
    stream = BatchStream(lab, columns, tokens, emit or events.append, stop)
    ids, mask, _ = lab.batch([lab.chat_ids(c["history"]) for c in columns])
    layers = set().union(*vectors)
    lab.pushes = {l: torch.stack([v.get(l, torch.zeros_like(next(v[l] for v in vectors if l in v)))
                                 for v in vectors])[:, None, :] for l in layers}
    try:
        torch.manual_seed(0)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
        with torch.inference_mode():
            lab.model.generate(ids, attention_mask=mask, max_new_tokens=tokens, repetition_penalty=1.1,
                               streamer=stream, stopping_criteria=StoppingCriteriaList([stream]), **kw)
    finally:
        lab.pushes = {}
    return events
