import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

from lab import Lab
from map import map_forward, map_layer, main


class GemmaMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        config = Gemma4Config(text_config=Gemma4TextConfig(
            vocab_size=64, vocab_size_per_layer_input=64, hidden_size=16,
            intermediate_size=24, num_hidden_layers=4, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, global_head_dim=8,
            hidden_size_per_layer_input=8, num_kv_shared_layers=2,
            use_double_wide_mlp=True,
            layer_types=['sliding_attention', 'full_attention'] * 2))
        model = Gemma4ForConditionalGeneration(config).eval()
        with patch('lab.AutoConfig.from_pretrained', return_value=config), \
             patch('lab.AutoTokenizer.from_pretrained', return_value=SimpleNamespace(pad_token_id=0, decode=lambda ids: str(ids[0]))), \
             patch('lab.Gemma4ForConditionalGeneration.from_pretrained', return_value=model), \
             patch('lab.device', return_value='cpu'):
            cls.lab = Lab('tiny-gemma')
        cls.batch = cls.lab.batch([[2, 4, 5], [2, 8]])
        cls.saved = cls.lab.layer_outputs(*cls.batch)
        cls.norms = {i: h[cls.batch[1].bool()].norm(dim=-1).median().item() for i, h in cls.saved.items()}
        with torch.inference_mode():
            cls.base = cls.lab.last_logprobs(*cls.batch)

    def test_widths_and_forbidden_restart(self):
        self.assertEqual(self.lab.neurons_per_layer, [24, 24, 48, 48])
        self.assertEqual(json.loads(json.dumps(self.lab.n_neurons)), [24, 24, 48, 48])
        with self.assertRaisesRegex(ValueError, 'full forwards'):
            with self.lab.start_after(2, self.saved[2]):
                pass

    def test_map_matches_plain_full_model(self):
        args = SimpleNamespace(controls=1, per_batch=2, dose=0.6, k=12)
        for layer in (0, 1, 2, 3):
            arrays = map_layer(self.lab, layer, self.batch, self.base, self.saved, self.norms, args, 2)
            self.assertEqual(arrays['up_tok'].shape, (2, 12))
            self.assertEqual(arrays['null_up_tok'].shape, (1, 12))
            for n in (0, 1):
                for name, sign in [('up', 1), ('down', -1)]:
                    vector = self.lab.push_vector(layer, self.lab.col(layer, n), sign * args.dose, self.norms)
                    layer_mod = self.lab.layers[layer]
                    after_norm = vector * layer_mod.post_feedforward_layernorm.weight.float()
                    after_norm = after_norm * vector.norm() / after_norm.norm() / layer_mod.layer_scalar.float()
                    h = layer_mod.post_feedforward_layernorm.register_forward_hook(
                        lambda m, inp, out: out + after_norm.to(out.dtype))
                    try:
                        ids, mask, pos = self.batch
                        with torch.inference_mode():
                            lp = self.lab.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                                use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
                    finally:
                        h.remove()
                    delta = lp - self.base
                    np.testing.assert_array_equal(arrays[f'{name}_tok'][n], delta.mean(0).topk(args.k).indices)
                    kl = (lp.exp() * delta).sum(-1).mean().item()
                    self.assertAlmostEqual(float(arrays[f'{name}_kl'][n]), kl, delta=max(1e-7, kl * 1e-3))

    def test_full_widths_and_invalid_limit(self):
        args = SimpleNamespace(controls=0, per_batch=3, dose=0.6, k=4)
        for layer, width in [(1, 24), (2, 48)]:
            arrays = map_layer(self.lab, layer, self.batch, self.base, self.saved, self.norms, args, None)
            self.assertEqual(arrays['down_kl'].shape, (width,))
            self.assertEqual(arrays['logit_lens'].shape, (width, 2, 8))
            self.assertEqual(arrays['null_up_kl'].shape, (0,))
        with self.assertRaisesRegex(ValueError, 'limit'):
            map_layer(self.lab, 1, self.batch, self.base, self.saved, self.norms, args, 25)

    def test_singleton_full_forward_collects_each_spread(self):
        args = SimpleNamespace(controls=0, per_batch=2, dose=0.6, k=4, full_forward=True)
        self.lab.is_gemma = False
        try:
            arrays = map_layer(self.lab, 1, self.batch, self.base, self.saved, self.norms, args, 2)
            for neuron in (0, 1):
                for name, sign in [('up', 1), ('down', -1)]:
                    vector = self.lab.push_vector(1, self.lab.col(1, neuron), sign * args.dose, self.norms)
                    self.lab.pushes = {1: vector}
                    states = self.lab.layer_outputs(*self.batch)
                    for layer in (1, 2, 3):
                        expected = ((states[layer][:, -1] - self.saved[layer][:, -1]).norm(dim=-1) /
                                    self.norms[layer]).mean().item()
                        self.assertAlmostEqual(float(arrays[f'{name}_spread'][neuron, layer]), expected,
                                               delta=max(1e-7, expected * 1e-3))
        finally:
            self.lab.is_gemma = True
            self.lab.pushes = {}

    def test_find_search_handles_variable_width_gradients(self):
        import find

        args = SimpleNamespace(pairs=None, words='word', layers='1-2', dose=0.6, top=4, controls=1)
        with patch.object(self.lab, 'prompts', return_value=[]), \
             patch.object(self.lab, 'norm_batch', return_value=self.batch), \
             patch('find.word_ids', return_value=[4]):
            result = find.search(self.lab, args)
        self.assertEqual(len(result['rows']), 4)
        self.assertIsNone(self.lab.zeros)
        self.assertEqual(self.lab.pushes, {})
        for row in result['rows']:
            self.assertIn(row['layer'], (1, 2))
            self.assertLess(row['n'], self.lab.neuron_count(row['layer']))
            self.assertTrue(np.isfinite(row['guess']))

    def test_push_restored_on_error(self):
        before = {3: torch.zeros(16)}
        self.lab.pushes = before
        try:
            with patch.object(self.lab, 'last_logprobs', side_effect=RuntimeError('test')):
                with self.assertRaisesRegex(RuntimeError, 'test'):
                    map_forward(self.lab, 1, self.batch, self.saved, torch.zeros(2, 1, 16))
            self.assertIs(self.lab.pushes, before)
        finally:
            self.lab.pushes = {}

    def test_gemma_main_skips_qwen_kernels_and_writes_widths(self):
        with TemporaryDirectory() as folder, \
             patch('sys.argv', ['map.py', '--model', 'tiny-gemma', '--layers', '2', '--limit', '2', '--out', folder]), \
             patch('map.Lab', return_value=self.lab), \
             patch.object(self.lab, 'dev', 'cuda'), \
             patch.object(self.lab, 'batch', return_value=self.batch), \
             patch('map.map_layer', return_value={'up_kl': np.zeros(2), 'null_up_kl': np.ones(1)}), \
             patch('map.frame_ids', return_value=[[2, 4, 5], [2, 8]]), \
             patch('map.map_kernels') as kernels:
            main()
            kernels.assert_not_called()
            meta = json.loads((Path(folder) / 'tiny-gemma/meta.json').read_text())
            self.assertEqual(meta['neurons'], [24, 24, 48, 48])
            self.assertTrue((Path(folder) / 'tiny-gemma/L2_first2.npz').exists())
            meta['dtype'] = 'torch.bfloat16'
            (Path(folder) / 'tiny-gemma/meta.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'fresh --out'):
                main()


if __name__ == '__main__':
    unittest.main()
