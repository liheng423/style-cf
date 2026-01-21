import torch.nn as nn
from models.agent import Agent
from models.benchmarks import IDM
from src.models.configs import test_config
import torch

def init_idm_model(model: IDM, model_config: dict, model_state: list):
    
    idm = model_config["model_name"](model_state)

    idm.to(test_config["device"])

    


def init_test_model(model: nn.Module, model_config: dict, model_state: dict):
    
    model = model_config["model_name"](model_config)
    if getattr(model, "use_dummy_style"): model.use_dummy_style = False 
    
    model.to(test_config["device"])

    ## load state dict 
    model.load_state_dict(model_state)
    return model


def style_data_split():
    pass

def agent(model, model_config, dt):

    horizon_len = model_config["pred_len"]
    historic_step = model_config["seq_len"]
    scaler = model_config["scaler"]()
    pred_func = model_config["pred_func"]
    mask = model_config["mask"]
    
    agent = Agent(model, dt, horizon_len, historic_step, scaler, pred_func, mask)
    agent._update_train_series = model_config["update_func"](agent)
    agent._concat = model_config["concat"]
    
    return agent

