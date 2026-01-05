import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import unittest
import torch
from tensordict.tensordict import TensorDict
from src.models.utils import stack_dim, stack_name



class TestUtils(unittest.TestCase):

    def test_stack_dim_basic(self):
        td1 = TensorDict({"a": torch.tensor([1, 2]), "b": torch.tensor([10, 20])}, batch_size=[2])
        td2 = TensorDict({"a": torch.tensor([3, 4]), "b": torch.tensor([30, 40])}, batch_size=[2])
        
        stacked_td = torch.concat([td1, td2], dim=0)

        self.assertTrue(torch.equal(stacked_td["a"], torch.tensor([[1, 2], [3, 4]])))
        self.assertTrue(torch.equal(stacked_td["b"], torch.tensor([[10, 20], [30, 40]])))
        self.assertEqual(stacked_td.batch_size, [2, 2])

