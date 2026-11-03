import json
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import label_fire
from label_fire import LabelFireError, auc, capture, check_texts, peak, permutation_p, run
from lab import Lab

WIDTH = 40
PAD = 0
HOT = 1
COLD = 2
START = 3
SPIKE = 4
GOOD = [f"start hot word {i}" for i in range(5)]
MISS = [f"start plain word {i}" for i in range(5)]


class Tok:
    pad_token_id = PAD
    vocab = {"hot": HOT, "cold": COLD, "start": START, "spike": SPIKE}

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[self.vocab.get(w, 10 + hash_word(w)) for w in text.split()])

    def decode(self, ids):
        return next((w for w, i in self.vocab.items() if i == ids[0]), "other")


def hash_word(word):
    return sum(ord(c) for c in word) % 50


def fake_lab(all_read_hot=False):
    lab = object.__new__(Lab)
    lab.model_id = "fake"
    lab.tok = Tok()
    lab.dev = "cpu"
    table = torch.randn(70, WIDTH, generator=torch.Generator().manual_seed(1)) * 0.1
    table[:, :2] = 0
    table[[HOT, COLD], 2:] = 0
    table[HOT, 0] = 5
    if all_read_hot:
        table[HOT] = 5
    table[COLD, 1] = -5
    table[PAD, 0] = 99
    table[SPIKE, 0] = 50
    down = torch.nn.Linear(WIDTH, 3, bias=False)
    down.requires_grad_(False)
    lab.layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=down))]
    lab.n_layers = 1
    lab.neurons_per_layer = [WIDTH]
    lab.lm = lambda input_ids, attention_mask, position_ids, use_cache: down(table[input_ids])
    return lab


def writer(examples=GOOD, near=MISS):
    return lambda label: {"examples": examples, "near_misses": near}


class ScoreTest(unittest.TestCase):
    def test_auc_counts_ties_as_half(self):
        self.assertEqual(auc([2, 2], [1, 1]), 1)
        self.assertEqual(auc([1, 1], [2, 2]), 0)
        self.assertEqual(auc([1, 1], [1, 1]), 0.5)

    def test_permutation_p_is_exact(self):
        self.assertAlmostEqual(permutation_p([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]), 1 / 252)


class CheckTextsTest(unittest.TestCase):
    def test_rejects_wrong_shapes(self):
        for bad in ({"examples": GOOD}, {"examples": GOOD[:4], "near_misses": MISS},
                    {"examples": GOOD, "near_misses": [GOOD[0], *MISS[1:]]},
                    {"examples": GOOD, "near_misses": ["", *MISS[1:]]}, []):
            with self.assertRaises(LabelFireError):
                check_texts(bad)


class CaptureTest(unittest.TestCase):
    def test_reads_down_proj_input_and_drops_padding(self):
        got = capture(fake_lab(), 0, ["start hot", "start plain word hot word"])
        self.assertEqual([acts.shape for acts, _ in got], [torch.Size([2, WIDTH]), torch.Size([5, WIDTH])])
        self.assertEqual(got[0][0][1, 0].item(), 5)
        self.assertNotIn(99, got[0][0][:, 0].tolist())

    def test_removes_its_hook(self):
        lab = fake_lab()
        capture(lab, 0, ["start hot"])
        self.assertFalse(lab.layers[0].mlp.down_proj._forward_pre_hooks)

    def test_first_token_does_not_count(self):
        got = capture(fake_lab(), 0, ["spike plain word"])[0]
        self.assertLess(peak(got, 0, 1)[0], 50)


class RunTest(unittest.TestCase):
    def test_true_label_passes(self):
        got = run(fake_lab(), "heat", 0, 0, 1, writer())
        self.assertEqual((got["verdict"], got["auc"]), ("pass", 1.0))
        self.assertEqual((got["example_fire_rate"], got["near_miss_fire_rate"]), (1, 0))
        self.assertEqual(got["examples"][0]["top_token"], "hot")
        self.assertAlmostEqual(got["p"], 1 / 252, places=3)

    def test_swapped_sets_fail(self):
        got = run(fake_lab(), "heat", 0, 0, 1, writer(MISS, GOOD))
        self.assertEqual((got["verdict"], got["auc"]), ("fail", 0.0))

    def test_negative_pole_reads_negative_values(self):
        examples = [t.replace("hot", "cold") for t in GOOD]
        self.assertEqual(run(fake_lab(), "chill", 0, 1, -1, writer(examples))["verdict"], "pass")
        self.assertEqual(run(fake_lab(), "chill", 0, 1, 1, writer(examples))["example_fire_rate"], 0)

    def test_small_separation_is_not_a_pass(self):
        examples = GOOD[:3] + MISS[:2]
        near = GOOD[3:] + MISS[2:]
        self.assertEqual(run(fake_lab(), "heat", 0, 0, 1, writer(examples, near))["verdict"], "weak")

    def test_texts_that_separate_every_neuron_are_not_a_pass(self):
        got = run(fake_lab(all_read_hot=True), "heat", 0, 0, 1, writer())
        self.assertEqual((got["auc"], got["verdict"]), (1.0, "weak"))

    def test_writer_sees_only_the_label(self):
        seen = []
        run(fake_lab(), "heat", 0, 0, 1, lambda *a, **k: seen.append((a, k)) or writer()("heat"))
        self.assertEqual(seen, [(("heat",), {})])

    def test_unknown_neuron_is_a_fixable_error_before_writing(self):
        called = []
        with self.assertRaises(LabelFireError) as e:
            run(fake_lab(), "heat", 0, WIDTH, 1, lambda label: called.append(label))
        self.assertEqual((e.exception.code, called), ("bad_neuron", []))


class ClaudeWriterTest(unittest.TestCase):
    def test_runs_with_no_tools_from_an_empty_folder(self):
        reply = {"structured_output": {"examples": GOOD, "near_misses": MISS}}
        done = subprocess.CompletedProcess([], 0, json.dumps(reply), "")
        with mock.patch.object(subprocess, "run", return_value=done) as call:
            self.assertEqual(label_fire.claude_writer("m")("heat"), reply["structured_output"])
        command, options = call.call_args.args[0], call.call_args.kwargs
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertIn("'heat'", command[2])
        self.assertTrue(options["cwd"].startswith(tempfile.gettempdir()))

    def test_failure_is_a_fixable_error(self):
        done = subprocess.CompletedProcess([], 1, "", "no auth")
        with mock.patch.object(subprocess, "run", return_value=done), self.assertRaises(LabelFireError) as e:
            label_fire.claude_writer("m")("heat")
        self.assertEqual(e.exception.code, "writer_failed")


if __name__ == "__main__":
    unittest.main()
