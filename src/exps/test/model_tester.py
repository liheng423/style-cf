import torch.nn as nn
from src.exps.agent import Agent
from src.exps.models.idm import IDM
from src.exps.configs import test_config
import torch

def init_idm_model(model: IDM, model_config: dict, model_state: list):
    
    idm = model_config["model_name"](model_state)

    idm.to(test_config["device"])
    return idm

def init_test_model(model: nn.Module, model_config: dict, model_state: dict):
    
    model = model_config["model_name"](model_config)
    if getattr(model, "use_dummy_style"): model.use_dummy_style = False 
    
    model.to(test_config["device"])

    ## load state dict 
    model.load_state_dict(model_state)
    return model


def style_data_split():
    pass

def agent(model: nn.Module, horizon_len: int, historic_step: int, agent_config: dict, dt: float, start_step: int = 0) -> Agent:

    scaler = agent_config["scaler"]()
    pred_func = agent_config["pred_func"]
    mask = agent_config["mask"]
    
    agent = Agent(model, dt, horizon_len, historic_step, scaler, pred_func, mask)
    agent._update_train_series = agent_config["update_func"](agent)
    agent._concat = agent_config["concat"]
    
    return agent

