from src.exps.utils.utils import SliceableTensorDict
from src.exps.utils.utils import load_zen_data
from src.exps.agent import Agent
from src.exps.models.idm import IDM, idm_concat, idm_update_train_series
import os
import sys
import unittest
import torch

from src.stylecf.schema import TensorNames


class TestIDMAgent(unittest.TestCase):
    def test_idm_agent_predict(self):
        params = [24.682408, 1.678458, 1.375292, 1.689426, 2.461880]
        dt = 0.1
        horizon_len = 1
        historic_step = 1

        idm = IDM(params, use_torch=True)
        idm_simulator = Agent(idm, dt, horizon_len, historic_step, scaler=None, start_timestep=0)
        idm_simulator._update_train_series = idm_update_train_series(idm_simulator)
        idm_simulator._concat = idm_concat

        total_steps = 8
        t = torch.arange(total_steps, dtype=torch.float32)

        self_v = torch.full((total_steps,), 10.0)
        leader_v = torch.full((total_steps,), 12.0)
        self_x = self_v * dt * t
        leader_x = self_x + 20.0
        self_a = torch.zeros(total_steps)
        leader_a = torch.zeros(total_steps)

        self_traj_full = torch.stack([self_x, self_v, self_a], dim=1)
        leader_traj_full = torch.stack([leader_x, leader_v, leader_a], dim=1)
        spacing = leader_x - self_x
        x_full_data = torch.stack([self_v, leader_v, spacing], dim=1).rename(TensorNames.T, TensorNames.F)
        x_full = SliceableTensorDict({TensorNames.INPUTS: x_full_data}, batch_size=[])
        pred = idm_simulator.predict(
            x_full,
            self_traj_full,
            leader_traj_full,
            pred_func=lambda m, d, *args: m(d),
            mask=lambda x, *args: x,
        )

        self.assertEqual(pred.shape, self_traj_full.shape)
        self.assertTrue(torch.isfinite(pred).all())
        mse = torch.mean((pred - self_traj_full) ** 2).item()
        self.assertLess(mse, 1) # Do not deviate too much
