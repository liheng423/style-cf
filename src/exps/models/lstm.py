import torch
import torch.nn as nn
from typing import List
from tensordict import TensorDict
from torch.nn import functional as F

from ..agent import Agent
from ..utils.utils import SliceableTensorDict, td_cat
from ..utils.utils_namebuilder import _build_name_dict
from ...schema import FEATNAMES as FEAT
from ...schema import CFNAMES as CF



# ========== LSTM Model ========== #

@staticmethod
def param_init(model, config, init_func=nn.init.kaiming_normal_):

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init_func(m.weight, mode='fan_in', nonlinearity=config["activate_name"])
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init_func(param, mode='fan_in', nonlinearity=config["activate_name"])
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

## modules ##

class RegularizeOutput(nn.Module):
    def __init__(self, map_range, activation='tanh'):
        """
        Args:
            map_range (tuple): (min_val, max_val) specifying the desired output range.
            activation (str): 'tanh' or 'sigmoid' to define the scaling function.
        """
        super().__init__()
        self.min_val, self.max_val = map_range
        self.scale = (self.max_val - self.min_val) / 2 if activation == 'tanh' else (self.max_val - self.min_val)
        self.shift = (self.max_val + self.min_val) / 2 if activation == 'tanh' else self.min_val
        
        if activation == 'tanh':
            self.regularize = nn.Tanh()
        elif activation == 'sigmoid':
            self.regularize = nn.Sigmoid()
        else:
            raise ValueError("Activation must be 'tanh' or 'sigmoid'")
    
    
    def forward(self, x):
        return self.shift + self.scale * self.regularize(x)


class LinearModule(nn.Module):

    def __init__(self, in_dim, out_dim, config):
        super().__init__()

        self.module = nn.Sequential(
            nn.BatchNorm1d(in_dim) if config["batch_norm"] else nn.Identity(),  # Apply BatchNorm before Linear
            nn.Linear(in_dim, out_dim),   # Reduced size
            config["activation_func"],                       
            nn.Dropout(config["dropout"]) if config["dropout"] > 0 else nn.Identity() ,
        )
    
    def forward(self, x):
        return self.module(x)
    



class Module_State_LSTM(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Simplified LSTM branch (Reducing hidden size and layers)
        self.lstm = nn.LSTM(
            input_size=config["num_state_feature"],
            hidden_size=64,  
            num_layers=1,    
            batch_first=True
        )

        self.module = nn.Sequential(
            LinearModule(64, 32, config))       



    def forward(self, x):
        _, (lstm_out, _) = self.lstm(x)
        lstm_out = lstm_out.permute(1, 0, 2)
        return self.module(lstm_out)

class Module_Output(nn.Module):

    def __init__(self, in_dim, config):
        super().__init__()
        self.out_module = nn.Sequential(
            nn.BatchNorm1d(in_dim) if config["batch_norm"] else nn.Identity(),
            nn.Linear(in_dim, config["pred_step"]),  # Final output layer
            RegularizeOutput(config["regular_range"], config["regular_func"]) if config["regular_output"] else nn.Identity()
        )



    def forward(self, x):
        return self.out_module(x)
    

class Module_TemporalAttentionEncoder(nn.Module):
    def __init__(self, config):
        super(Module_TemporalAttentionEncoder, self).__init__()
        self.key_layer = nn.Linear(config["num_feature"], 128)
        self.query = nn.Parameter(torch.randn(128))  # learnable global query
        self.output_layer = nn.Linear(config["num_feature"], 256)

    def forward(self, x):
        """
        x: Tensor of shape (batch, time, num_feat)
        Returns: Tensor of shape (batch, output_dim)
        """
        # Project inputs to key space: (batch, time, hidden)
        keys = torch.tanh(self.key_layer(x))

        # Compute attention scores via dot product with global query
        attn_scores = torch.matmul(keys, self.query)  # shape: (batch, time)

        # Softmax over time dimension
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, time, 1)

        # Weighted sum of original inputs
        weighted_sum = torch.sum(attn_weights * x, dim=1)  # (batch, num_feat)

        # Project to desired output dim
        out = self.output_layer(weighted_sum)  # (batch, output_dim)
        return out



class Module_CF_LSTM(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Simplified LSTM branch (Reducing hidden size and layers)
        self.lstm_micro = nn.LSTM(
            input_size=config["num_feature"],
            hidden_size=128,  
            num_layers=2,    
            batch_first=True,
            dropout=0,
            bidirectional=config["bidirectional"]
        )

        # Fusion layer with reduced size
        self.fc = nn.Sequential(
            LinearModule(256, 128, config),
            LinearModule(128, 64, config),
            LinearModule(64, 32, config),
        )



    def forward(self, x_micro):
        # Process LSTM output
        _, (lstm_out, _) = self.lstm_micro(x_micro)
        lstm_out = lstm_out.permute(1, 0, 2)
        fc_out = self.fc(lstm_out.flatten(start_dim=1))
        return fc_out

class CF_LSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.cf_lstm = Module_CF_LSTM(config)
        self.out_module = Module_Output(32, config)

        param_init(self, config) 

    def forward(self, x):
        return self.out_module(self.cf_lstm(x))


# ======================================= #
def lstm_concat(tensor_list: List[TensorDict]):
    """
        tensor_list: List[TensorDict], no style token, thus reduce to normal concat.
    """
    return td_cat(tensor_list, dim=0)

def lstm_update_func(simulator: Agent, feature_dict: dict[str, List[str]]):
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
        td_cls = type(train_series)
        return td_cls(out, batch_size=train_series.batch_size, names=train_series.names)
    
    return _update_train_series
