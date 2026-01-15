from ast import List
from typing import override
from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== IDM Model ========== #
class IDM(nn.Module):
    def __init__(self, params, use_torch=False):
        
        self.use_torch = use_torch

        if use_torch:
            super(IDM, self).__init__()  
            self.v0 = nn.Parameter(torch.tensor(params[0], dtype=torch.float32))
            self.s0 = nn.Parameter(torch.tensor(params[1], dtype=torch.float32))
            self.T = nn.Parameter(torch.tensor(params[2], dtype=torch.float32))
            self.a = nn.Parameter(torch.tensor(params[3], dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(params[4], dtype=torch.float32))
        else:
            self.v0, self.s0, self.T, self.a, self.b = params

        self.sigma = 4  # Fixed exponent

    def desired_s(self, v_this, delta_v_i):
        if self.use_torch:
            return self.s0 + torch.max(torch.tensor(0.0, device=v_this.device), 
                                   v_this * self.T + v_this * delta_v_i / (2 * torch.sqrt(self.a * self.b)))
        else:
            return self.s0 + max(0, v_this * self.T + v_this * delta_v_i / (2 * torch.sqrt(self.a * self.b)))

    def predict(self, v_this, v_front, s_this):
        """
        This function does the same thing as forward function
        But with a normal interface
        """
        delta_v_i = v_this - v_front
        return self.a * (1 - (v_this / self.v0) ** self.sigma - (self.desired_s(v_this, delta_v_i) / s_this) ** 2)

    def forward(self, X, *args):
        """
        Args:
        X (torch.Tensor): size (batch X [v_self, v_leader, spacing]), training data input.
        """
        v_this = X[:, 0]
        v_front = X[:, 1]
        s_this = X[:, 2]
        delta_v_i = v_this - v_front
        return self.a * (1 - (v_this / self.v0) ** self.sigma - (self.desired_s(v_this, delta_v_i) / s_this) ** 2)

@staticmethod
def idm_update_train_series(simulator):

    def _update_train_series(train_series: torch.Tensor, self_movements: torch.Tensor, leader_movements: torch.Tensor):
        """
        Args:
            X (torch.Tensor): size (1, [v_self, v_leader, spacing]), training data input.
            self_movements (np.array): size (1, [x_self, v_self, a_self])
            leader_movements (np.array): size (1, [x_self, v_self, a_self])
        """
        train_series = train_series.inputs[0]
        train_series[:, 0] = self_movements[:, 1]
        train_series[:, 2] = leader_movements[:, 0] - self_movements[:, 0]

        return torch.tensor(train_series)

    return _update_train_series

@override
def idm_concat(tensor_list: List[TensorDict]):
    """
        tensor_list: List[TensorDict], no style token, thus reduce to normal concat.
    """
    return torch.concat(tensor_list, dim=0)


# ========== End of IDM Model ========== #

# ========== LSTM Model ========== #


@override
def lstm_concat(tensor_list: List[TensorDict]):
    """
        tensor_list: List[TensorDict], no style token, thus reduce to normal concat.
    """
    return torch.concat(tensor_list, dim=0)

