import json

import pytest
import torch
import urllib.error

import steering_score as ss
from config import FRAMES, HELD_OUT, PROMPTS, SCORE_PROMPTS
from lab import random_vectors, steer_vectors

ENTRY = dict(id='a', model='Qwen/Qwen3.5-2B', layer=17, n=5244, dose=.6, name='Evil', note='0.5 L17 secret')
COMBO = dict(id='b', model='Qwen/Qwen3.5-2B', name='Evil God', pushes=[dict(layer=17, n=604, dose=.5), dict(layer=17, n=5244, dose=.4)])


class FakeGenerate:
    def __init__(self):
        self.calls = []

    def __call__(self, model, pushes, seed, prompts, tokens):
        self.calls.append((model, pushes, seed, tokens))
        word = 'plain' if seed is not None else 'evil'
        return [dict(text=f'{word} reply {i}', stopped='eos') for i in range(len(prompts))]


class FakeJudge:
    name = 'fake'

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after

    def __call__(self, concept, prompt, reply):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise ss.ScoreError('judge_failed', 'down', 'retry')
        self.calls.append((concept, prompt, reply))
        hit = concept == 'Evil' and reply.startswith('evil')
        return dict(concept=2 if hit else 0, instruction=2, fluency=2)


def test_scores_steered_against_control(tmp_path):
    generate, judge, path = FakeGenerate(), FakeJudge(), tmp_path / 'r.json'
    trial = ss.score_entry(ENTRY, generate, judge, path)
    assert [c[2] for c in generate.calls] == [None, int(ss.run_key(ENTRY)[:8], 16)]
    assert generate.calls[0][1] == [dict(layer=17, n=5244, dose=.6)] and generate.calls[0][3] == 128
    assert len(judge.calls) == 20
    assert trial['score']['steered'] == dict(concept=2, instruction=2, fluency=2, overall=2)
    assert trial['score']['control'] == dict(concept=0, instruction=2, fluency=2, overall=0)
    assert trial['score']['difference'] == 2 and trial['score']['interval'] == (2, 2)
    saved = json.loads(path.read_text())['runs'][ss.run_key(ENTRY)]
    assert saved['trials']['fake|Evil']['score'] == json.loads(json.dumps(trial['score']))
    assert len(saved['replies']['control']) == 10


def test_judge_is_blind_and_shuffled(tmp_path):
    judge = FakeJudge()
    ss.score_entry(ENTRY, FakeGenerate(), judge, tmp_path / 'r.json')
    assert {c[0] for c in judge.calls} == {'Evil'}
    words = [c[2].split()[0] for c in judge.calls]
    assert words != sorted(words) and words != sorted(words, reverse=True)
    assert not any('secret' in part or '5244' in part or '0.6' in part for call in judge.calls for part in call)


def test_harmonic_mean_punishes_any_zero():
    assert ss.overall(dict(concept=2, instruction=0, fluency=2)) == 0
    assert ss.overall(dict(concept=2, instruction=1, fluency=2)) == pytest.approx(1.5)


def test_resumes_without_regenerating_or_regrading(tmp_path):
    generate, path = FakeGenerate(), tmp_path / 'r.json'
    with pytest.raises(ss.ScoreError):
        ss.score_entry(ENTRY, generate, FakeJudge(fail_after=7), path)
    run = json.loads(path.read_text())['runs'][ss.run_key(ENTRY)]
    assert sum(len(g) for g in run['trials']['fake|Evil']['grades'].values()) == 7
    judge = FakeJudge()
    trial = ss.score_entry(ENTRY, generate, judge, path)
    assert len(judge.calls) == 13 and len(generate.calls) == 2 and trial['score']['n'] == 10
    again = FakeJudge()
    ss.score_entry(ENTRY, generate, again, path)
    assert again.calls == [] and len(generate.calls) == 2


def test_new_dose_regenerates_and_new_name_only_regrades(tmp_path):
    generate, path = FakeGenerate(), tmp_path / 'r.json'
    ss.score_entry(ENTRY, generate, FakeJudge(), path)
    judge = FakeJudge()
    ss.score_entry(ENTRY | dict(name='Villain'), generate, judge, path)
    assert len(generate.calls) == 2 and len(judge.calls) == 20
    ss.score_entry(ENTRY | dict(dose=.4), generate, FakeJudge(), path)
    assert len(generate.calls) == 4


def test_combo_keeps_all_pushes(tmp_path):
    generate = FakeGenerate()
    ss.score_entry(COMBO, generate, FakeJudge(), tmp_path / 'r.json')
    assert generate.calls[1][1] == sorted(COMBO['pushes'], key=lambda p: p['n'])


def test_refuses_unnamed_and_doseless(tmp_path):
    with pytest.raises(ss.ScoreError) as e:
        ss.score_entry(ENTRY | dict(name='L17 N5244'), FakeGenerate(), FakeJudge(), tmp_path / 'r.json')
    assert e.value.code == 'no_label'
    with pytest.raises(ss.ScoreError) as e:
        ss.score_entry(ENTRY | dict(dose=None), FakeGenerate(), FakeJudge(), tmp_path / 'r.json')
    assert e.value.code == 'no_dose'


def test_rejects_bad_grades(tmp_path):
    class Bad(FakeJudge):
        def __call__(self, *_):
            return dict(concept=3, instruction=2, fluency=2)
    with pytest.raises(ss.ScoreError) as e:
        ss.score_entry(ENTRY, FakeGenerate(), Bad(), tmp_path / 'r.json')
    assert e.value.code == 'bad_grade'


def test_swapped_labels_never_keep_their_own():
    rows = [ENTRY | dict(n=i, name=name) for i, name in enumerate(['A', 'A', 'B', 'C', 'L1 N2'])]
    swaps = ss.swapped_labels(rows)
    assert len(swaps) == 4 and 'L1 N2' not in swaps.values()
    for row in rows[:4]:
        assert swaps[repr(ss.library.identity(row))] != row['name']
    with pytest.raises(ss.ScoreError):
        ss.swapped_labels([ENTRY, ENTRY | dict(n=1)])


def library_file(tmp_path, count=6):
    names = ['Evil'] + [f'Other {i}' for i in range(count - 1)]
    path = tmp_path / 'library.json'
    path.write_text(json.dumps([ENTRY | dict(id=str(i), n=i, name=n) for i, n in enumerate(names)]))
    return path


def test_cli_swapped_reuses_replies_and_reports_calibration(tmp_path, capsys):
    generate, judge = FakeGenerate(), FakeJudge()
    code = ss.main(['--all', '--swapped', '--library', str(library_file(tmp_path)), '--results', str(tmp_path / 'r.json')], generate, judge)
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out['ok'] and len(out['entries']) == 6
    assert len(generate.calls) == 12 and len(judge.calls) == 240
    evil = out['entries'][0]
    assert evil['score']['steered']['overall'] == 2 and evil['swapped']['score']['steered']['overall'] == 0
    assert evil['swapped']['label'] != 'Evil'
    assert out['calibration']['n'] == 6 and out['calibration']['passes'] is False


def test_calibration_passes_when_true_labels_win():
    trial = lambda value: dict(score=dict(steered=dict(overall=value)))
    good = ss.calibration([(trial(1.5 + i / 10), trial(.5)) for i in range(5)])
    assert good['passes'] is True and good['difference'] > 0
    assert ss.calibration([(trial(1.5), trial(.5))])['passes'] is None


@pytest.mark.parametrize('argv,code', [
    ([], 'no_entry'),
    (['--entry', 'Nobody'], 'bad_entry'),
])
def test_cli_errors_are_json(tmp_path, capsys, argv, code):
    assert ss.main(argv + ['--library', str(library_file(tmp_path)), '--results', str(tmp_path / 'r.json')], FakeGenerate(), FakeJudge()) == 1
    err = json.loads(capsys.readouterr().err)
    assert err['ok'] is False and err['code'] == code and err['fix']


def test_missing_key_fails_before_any_model(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    generate = FakeGenerate()
    assert ss.main(['--entry', 'Evil', '--library', str(library_file(tmp_path)), '--results', str(tmp_path / 'r.json')], generate) == 1
    assert json.loads(capsys.readouterr().err)['code'] == 'no_api_key' and generate.calls == []


def test_api_judge_forces_the_tool_and_retries_rate_limits():
    grades = dict(concept=2, instruction=1, fluency=0)
    calls, naps = [], []

    def post(body):
        calls.append(body)
        if len(calls) == 1:
            raise urllib.error.HTTPError('u', 429, 'slow', {}, None)
        return dict(content=[dict(type='tool_use', name='grade', input=grades)])
    judge = ss.ApiJudge('key', post=post, sleep=naps.append)
    assert judge('Evil', 'Hi', 'reply') == grades
    assert naps == [1] and calls[1]['tool_choice'] == dict(type='tool', name='grade')
    assert calls[1]['model'] == 'claude-haiku-5-5'
    assert '<concept>Evil</concept>' in calls[1]['messages'][0]['content']


def test_api_judge_stops_on_auth_error():
    def post(_):
        raise urllib.error.HTTPError('u', 401, 'no', {}, None)
    with pytest.raises(ss.ScoreError) as e:
        ss.ApiJudge('key', post=post, sleep=lambda _: None)('c', 'p', 'r')
    assert e.value.code == 'judge_failed' and '401' in e.value.error


def test_prompts_are_frozen_apart_from_other_sets():
    assert len(SCORE_PROMPTS) == len(set(SCORE_PROMPTS)) == 10
    used = set(PROMPTS) | set(HELD_OUT) | {q for q, _ in FRAMES}
    assert not used & set(SCORE_PROMPTS)


class FakeLab:
    dev = 'cpu'

    def __init__(self, width=32):
        layer = type('L', (), {'mlp': type('M', (), {'down_proj': type('D', (), {'out_features': width})})})
        self.layers = {3: layer, 7: layer}
        self.width = width

    def col(self, layer, n):
        return torch.ones(self.width) * (n + 1)

    def push_vector(self, layer, direction, dose, norms):
        return direction / direction.norm() * dose * norms[layer]


def test_random_control_matches_the_push_size():
    lab, norms = FakeLab(), {3: 10.0, 7: 20.0}
    rows = [(3, 1, .5), (7, 2, -.25)]
    steered, control = steer_vectors(lab, rows, norms), random_vectors(lab, rows, norms, 9)
    for layer in norms:
        assert control[layer].norm().item() == pytest.approx(steered[layer].norm().item())
        assert not torch.allclose(control[layer] / control[layer].norm(), steered[layer] / steered[layer].norm())
    again = random_vectors(lab, rows, norms, 9)
    assert torch.equal(again[3], control[3]) and not torch.equal(random_vectors(lab, rows, norms, 10)[3], control[3])
