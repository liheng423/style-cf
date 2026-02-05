from typing import Callable
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader
from exps.datahandle.datascalers import DataScaler
from src.exps.datahandle.databuilder import build_dataset
from src.exps.datahandle.dataset import LSTMDataset, IDMDataset, _fit_scaler, _transform
from src.exps.models.stylecf import StyleModel
from src.exps.agent import Agent
from src.exps.models.idm import IDM
from src.exps.configs import test_config, style_data_config, lstm_data_config, idm_calibration_config
from src.exps.train.model_trainer import build_style_loader
from src.schema import FEATNAMES as FEAT
import torch

def init_idm_model(model: IDM, model_config: dict, model_state: list):
    
    idm = model_config["model_name"](model_state)

    idm.to(test_config["device"])
    return idm

def init_test_model(model: StyleModel, model_config: dict, model_state: dict):
    
    model = model_config["model_name"](model_config)
    if getattr(model, "use_dummy_style"): model.use_dummy_style = False 
    
    model.to(test_config["device"])

    ## load state dict 
    model.load_state_dict(model_state)
    return model


def style_data_split():
    pass

def agent(model: nn.Module, data_config: dict, agent_config: dict, dt: float, scalers: dict[str, DataScaler], start_step: int = 0) -> Agent:


    pred_func = agent_config["pred_func"]
    mask = agent_config["mask"]
    horizon_len: int = data_config["horizon_len"]
    historic_step: int = data_config["historic_step"]
    
    agent = Agent(model, dt, horizon_len, historic_step, scalers, start_timestep=start_step)
    agent._update_train_series = agent_config["update_func"](agent)
    agent._concat = agent_config["concat"]
    
    return agent
