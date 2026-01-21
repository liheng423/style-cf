import torch
from typing import List
from tensordict import TensorDict

from src.exps.agent import Agent
from src.exps.utils.utils import SliceableTensorDict
from src.exps.utils.utils_namebuilder import _build_name_dict
from src.schema import FEATNAMES as FEAT
from src.schema import CFNAMES as CF



# ========== LSTM Model ========== #


def lstm_concat(tensor_list: List[TensorDict]):
    """
        tensor_list: List[TensorDict], no style token, thus reduce to normal concat.
    """
    return torch.concat(tensor_list, dim=0)

def lstm_update_train_series(simulator: Agent, feature_dict: dict[str, List[str]]):
    """
    Update LSTM input series in SliceableTensorDict form.
    """
    name_dict = _build_name_dict(feature_dict)

    def _update_train_series(train_series: SliceableTensorDict, self_movements: torch.Tensor, leader_movements: torch.Tensor):
        """
        Args:
            x_series: np.array (time, [delta_v, delta_x, v_self]]) By Default
            self_movements : np.array (time, [x_self, v_self, a_self])
            leader_movements : np.array (time, [x_self, v_self, a_self])
        """

        x_series = train_series[FEAT.INPUTS]
        x_names = x_series.names

        x_series = simulator.scalers[FEAT.INPUTS].inverse_transform(x_series)
        delta_x = leader_movements[:, 0] - self_movements[:, 0]
        delta_v = leader_movements[:, 1] - self_movements[:, 1]

        # update train series
        x_series[:, name_dict[FEAT.INPUTS][CF.SELF_V]] = self_movements[:, 1]
        x_series[:, name_dict[FEAT.INPUTS][CF.DELTA_X]] = delta_x
        x_series[:, name_dict[FEAT.INPUTS][CF.DELTA_V]] = delta_v

        x_series_scaled = torch.tensor(simulator.scalers[FEAT.INPUTS].transform(x_series)).float()
        if x_names is not None:
            x_series_scaled = x_series_scaled.refine_names(*x_names)

        out = {FEAT.INPUTS: x_series_scaled}
        return SliceableTensorDict(out, batch_size=train_series.batch_size, names=train_series.names)
    
    return _update_train_series
