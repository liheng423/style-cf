from typing import List, cast
from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import SliceableTensorDict, stack_name, td_cat
from ...schema import CFNAMES as CF, TensorNames



# default settings #

DEFAULT_PRED_FUNC = lambda model, data, *args: model(data)
DEFAULT_MASK = lambda x, *args: x

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
            term = v_this * self.T + v_this * delta_v_i / (2 * torch.sqrt(self.a * self.b))
            return self.s0 + torch.clamp_min(term, 0.0)
        else:
            return self.s0 + max(0, v_this * self.T + v_this * delta_v_i / (2 * torch.sqrt(self.a * self.b)))

    def predict(self, v_this, v_front, s_this):
        """
        This function does the same thing as forward function
        But with a normal interface
        """
        delta_v_i = v_this - v_front
        return self.a * (1 - (v_this / self.v0) ** self.sigma - (self.desired_s(v_this, delta_v_i) / s_this) ** 2)

    def forward(self, X: SliceableTensorDict, *args):
        """
        Args:
        X (SliceableTensorDict): size ((B), T, [v_self, v_leader, spacing]), training data input.
        """
        x = cast(torch.Tensor, X[TensorNames.INPUTS])  # get the underlying tensor
        v_this = x[... , 0]
        v_front = x[... , 1]
        s_this = x[... , 2]
        return self.predict(v_this, v_front, s_this)
        
@staticmethod
def idm_update_func(simulator):

    def _update_train_series(train_series: TensorDict, self_movements: torch.Tensor, leader_movements: torch.Tensor):
        """
        Args:
            train_series (TensorDict): Original training series.
            self_movements (torch.Tensor): size (T,  [x, v, a]) of the ego vehicle.
            leader_movements (torch.Tensor): size (T, [x, v, a]) of the leader vehicle.

        Returns:
            torch.Tensor: Updated training series.
        """
        x = cast(torch.Tensor, train_series[TensorNames.INPUTS])
        x[... , 0] = self_movements[... , 1]
        x[... , 2] = leader_movements[... , 0] - self_movements[... , 0]

        td_cls = type(train_series)
        return td_cls(
            {TensorNames.INPUTS: x},
            batch_size=train_series.batch_size,
            names=train_series.names,
        )

    return _update_train_series

def idm_concat(tensor_list: List[SliceableTensorDict]):
    """
        tensor_list: List[TensorDict] or List[torch.Tensor], no style token, thus reduce to normal concat.
    """
    if not tensor_list:
        raise ValueError("tensor_list must be non-empty")
    first = tensor_list[0]
    return stack_name(tensor_list, TensorNames.T)



# ========== End of IDM Model ========== #
