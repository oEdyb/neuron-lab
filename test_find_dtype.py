import unittest

import torch

from lab import Lab


class FindDtypeTest(unittest.TestCase):
    def test_bfloat16_push_retains_float32_gradient(self):
        lab = object.__new__(Lab)
        lab.zeros = [torch.zeros(4, requires_grad=True)]
        projection = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        projection.requires_grad_(False)
        projection.weight.fill_(0.25)
        projection.register_forward_pre_hook(lab._grad_hook(0))
        output = projection(torch.ones(1, 4, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().sum().backward()
        self.assertEqual(lab.zeros[0].grad.dtype, torch.float32)
        torch.testing.assert_close(lab.zeros[0].grad, torch.full((4,), 0.5))


if __name__ == '__main__':
    unittest.main()
