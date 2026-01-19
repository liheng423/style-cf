import torch.optim as optim
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
from src.models.dataset import StyledTransfollowerDataset
from src.schema import CFNAMES as CF
import torch.nn
from src.models.loss import StyleLoss
from src.models.style_cf import StyleTransformer, transformer_mask, style_pred_func 
best_model_path = f"models/best-model-{datetime.now():%Y%m%d-%H%M%S}.pth"


style_data_config = {
    "batch_size": 64,
    "train_data_ratio": 0.7,
    "seq_len": 60,# steps not seconds
    "label_len": 40,# steps not seconds
    "pred_len": 40, # steps not seconds
    "stride": 20, # steps not seconds
    "scaler": StandardScaler,
    "dataset": StyledTransfollowerDataset,
    "model_name": StyleTransformer,
    
    "x_groups": {
        "enc_x": {"features": [CF.SELF_V, CF.DELTA_X, CF.DELTA_V, CF.SELF_L, CF.LEAD_L], "transform": True},
        "dec_x": {"features": [CF.SELF_V, CF.LEAD_V], "transform": True},
        "style": {"features": [CF.REACT, CF.THW, CF.DELTA_X, CF.SELF_V], "transform": True},
    },
    "y_groups": {
        "y_seq": {"features": [CF.SELF_X, CF.SELF_V, CF.SELF_A, CF.DELTA_X, CF.LEAD_X]},
    },
}
 

style_train_config = {
    "num_epoch": 40,
    "max_norm": 10,
    "dt": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "best_model_path": best_model_path,

    "loss_func": StyleLoss.acc_spacing_mse,

    "optim": optim.Adam,
    "lr": 5e-4, # transfollower is 1e-4

    ### recursive ###
    "pred_func": style_pred_func,
    "mask": transformer_mask,
}

