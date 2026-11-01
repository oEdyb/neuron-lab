"""Reproduce the Ministral smoke benchmark and first steering experiments."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from config import MINISTRAL, FRAMES
from lab import Lab, generate_compare, steer_vectors
from find import search
from map import frame_ids, map_forward, map_layer


def benchmark(lab, out):
    batch = lab.batch(frame_ids(lab, FRAMES))
    saved = lab.layer_outputs(*batch)
    norms = {i: h[batch[1].bool()].float().norm(dim=-1).median().item() for i, h in saved.items()}
    with torch.inference_mode():
        base = lab.last_logprobs(*batch)
    report = dict(gpu=torch.cuda.get_device_name(), dtype=str(lab.model.lm_head.weight.dtype),
                  sequence_length=batch[0].shape[1], weights=lab.weights_id, revision=lab.revision,
                  tied=lab.model.lm_head.weight.data_ptr() == lab.lm.embed_tokens.weight.data_ptr(), checks=[], timings=[])
    for layer in (0, 13, 25):
        v = lab.push_vector(layer, lab.col(layer, 100), .6, norms)
        p = v[None, None].repeat(len(FRAMES), 1, 1)
        fast = map_forward(lab, layer, batch, saved, p)
        ref = map_forward(lab, layer, batch, saved, p, full_forward=True)
        fkl = (fast.exp() * (fast - base)).sum(-1).mean().item()
        rkl = (ref.exp() * (ref - base)).sum(-1).mean().item()
        top = (fast-base).mean(0).topk(24).indices.tolist()
        other = (ref-base).mean(0).topk(24).indices.tolist()
        row = dict(layer=layer, max_logprob_error=(fast-ref).abs().max().item(), kl=fkl,
                   reference_kl=rkl, top24_overlap=len(set(top)&set(other))/24)
        report['checks'].append(row)
        print('CHECK',json.dumps(row),flush=True)
        for per_batch in (2, 8, 16, 32):
            args = SimpleNamespace(controls=0, per_batch=per_batch, dose=.6, k=24)
            torch.cuda.synchronize()
            start=time.monotonic()
            arrays=map_layer(lab, layer, batch, base, saved, norms, args, 32)
            torch.cuda.synchronize()
            row=dict(layer=layer, per_batch=per_batch, seconds=time.monotonic()-start, neurons=32,
                     peak_gib=torch.cuda.max_memory_allocated()/2**30)
            report['timings'].append(row)
            print('TIMING',json.dumps(row),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n')


TRAITS = ['hostile', 'pirate', 'french', 'confident', 'poetic']


def find_traits(lab, out, layers=None):
    result = dict(model=lab.model_id, weights=lab.weights_id, revision=lab.revision,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, traits={})
    for trait in TRAITS:
        args = SimpleNamespace(pairs=f'pairs/ministral-{trait}.txt', words=None, layers=layers,
                               dose=.6, top=12, controls=4)
        start=time.monotonic()
        result['traits'][trait] = search(lab,args)
        result['traits'][trait]['seconds']=time.monotonic()-start
        print('FIND',trait,json.dumps(result['traits'][trait]),flush=True)
        out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


def reply(lab, history, steer):
    events=generate_compare(lab,[dict(id='reply',history=history)],[steer],tokens=160)
    done=next(event for event in events if event.get('done'))
    return dict(text=done['text'],stopped=done['stopped'])


def steering(lab, out, source, traits=None):
    finds=json.loads(Path(source).read_text())
    if set(finds['traits']) != set(TRAITS):
        raise ValueError('Find output is incomplete; wait for all five traits')
    norms=lab.stream_norms(*lab.norm_batch())
    prompt='Hi! Who are you?'
    questions=['My dog is called Bosse. Say hi to him!',
               'Solve 3x + 7 = 22. Keep it short.', 'What is my dog called?']
    result=dict(model=lab.model_id, weights=lab.weights_id, revision=lab.revision,
                generation=dict(do_sample=False, repetition_penalty=1.1,max_new_tokens=160,seed=0),
                norm_source='Lab.norm_batch', norms=norms, prompt=prompt, candidates=[], baseline={})
    result['baseline']['greeting']=reply(lab,[dict(role='user',content=prompt)],{})
    history=[]
    for question in questions:
        history.append(dict(role='user',content=question))
        response=reply(lab,history,{})
        history.append(dict(role='assistant',content=response['text'],stopped=response['stopped']))
    result['baseline']['conversation']=history
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for trait,find in finds['traits'].items():
        if traits and trait not in traits.split(','):
            continue
        eligible=[r for r in find['rows'] if r['beats_null']]
        for r in eligible[:2]:
            candidate=dict(trait=trait,layer=r['layer'],n=r['n'],sign=r['sign'],
                           find_real=r['real'],find_null=r['null'],greetings=[],conversation=[])
            for dose in (.4,.6,.8):
                pushes=[(r['layer'],r['n'],r['sign']*dose)]
                steer=steer_vectors(lab,pushes,norms)
                response=reply(lab,[dict(role='user',content=prompt)],steer)
                candidate['greetings'].append(dict(dose=r['sign']*dose,**response))
                print('STEER',trait,r['layer'],r['n'],r['sign']*dose,repr(response),flush=True)
            steer=steer_vectors(lab,[(r['layer'],r['n'],r['sign']*.6)],norms)
            history=[]
            for question in questions:
                history.append(dict(role='user',content=question))
                response=reply(lab,history,steer)
                history.append(dict(role='assistant',content=response['text'],stopped=response['stopped']))
            candidate['conversation']=history
            result['candidates'].append(candidate)
            out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['benchmark','find','steer'],default='benchmark')
    ap.add_argument('--out',default='out/ministral3-benchmark.json')
    ap.add_argument('--layers')
    ap.add_argument('--traits')
    ap.add_argument('--source',default='/cache/ministral3/find.json')
    args=ap.parse_args()
    torch.set_num_threads(8)
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    lab=Lab(MINISTRAL,True)
    if args.stage=='benchmark':
        benchmark(lab,out)
    elif args.stage=='find':
        find_traits(lab,out,args.layers)
    else:
        steering(lab,out,args.source,args.traits)
