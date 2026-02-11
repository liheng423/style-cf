import numpy as np
import torch

from src.exps.agent import Agent
from src.exps.configs import (
    data_filter_config,
    filter_names,
    idm_calibration_config,
    lstm_data_config,
    style_data_config,
    test_config,
)
from src.exps.datahandle import dataset
from src.exps.models.idm import IDM
from src.exps.test.model_tester import init_idm_model, init_test_model
from src.exps.train.model_trainer import build_style_loader
from src.exps.utils.utils import SliceableTensorDict, load_zen_data
from src.exps.utils.utils_namebuilder import _build_scaler_dict, build_name_list
from src.schema import CFNAMES as CF
from src.stylecf.schema import TensorNames
from src.exps.datahandle.filters import CFFilter

def _dataset(head=None):

    data_path = "~/datasets/DATA/ZenTraffic/ZenTraffic90kalman.npy"

    d = load_zen_data(data_path, rise=True, in_kph=False, kilo_norm=True)
    return d.head(head) if head is not None else d

def main():
    
    d = _dataset(1000)
   
    filter_list = build_name_list(filter_names, CFFilter) 

    _, train_loader, test_loader, _ = build_style_loader(d, filter_list, data_filter_config, data_config=style_data_config)
    
    

