import torch.nn as nn
from src.models.configs import test_config
import torch



def init_test_model(model: nn.Module, model_config: dict, model_state: dict):
    
    model = model_config["model_name"](model_config)
    if getattr(model, "use_dummy_style"): model.use_dummy_style = False 
    
    model.to(test_config["device"])

    ## load state dict 
    model.load_state_dict(model_state)
    return model


def style_data_split():
    pass

def agent():
    pass