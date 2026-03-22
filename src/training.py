# %load_ext autoreload
# %autoreload 2
# %% Imports

import os
import unittest
import numpy as np
import test

from .exps.train.model_trainer import build_style_dataset
from .exps.utils.utils import load_zen_data
from .exps.configs import style_data_config, style_train_config
from .schema import CFNAMES as CF
from .stylecf.schema import TensorNames

from .exps.train.model_trainer import train_stylecf
from .exps.models.stylecf import StyleTransformer
from .exps.configs import *
# %% Utils


def _dataset(head=None):

    data_path = "F:\DATA\ZenTraffic\ZenTraffic30kalman.npy"

    d = load_zen_data(data_path, rise=True, in_kph=False, kilo_norm=True)
    return d.head(head) if head is not None else d
# %% Train Style-cf

        
def test_style_train():
    
    d = _dataset()

    _, train_loader, test_loader, _ = build_style_dataset(d, filter_names, data_filter_config, data_config=style_data_config)

    model = train_stylecf(style_data_config, style_train_config, train_loader, test_loader)
    return model


test_style_train()
# %%
