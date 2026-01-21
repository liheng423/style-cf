from tarfile import data_filter
import torch.optim as optim
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
from src.exps.utils.utils import stack_name
from src.exps.models.idm import DEFAULT_MASK, DEFAULT_PRED_FUNC, idm_update_func, idm_update_train_series, idm_concat
from src.exps.datahandle.datascalers import DataScaler
from src.exps.datahandle.dataset import StyledTransfollowerDataset, LSTMDataset
from src.schema import CFNAMES as CF
import torch.nn as nn
from src.exps.loss import LossFunctions, StyleLoss, IDMLoss
from exps.models.stylecf import StyleTransformer, style_update_func, stylecf_mask, transformer_mask, style_pred_func 
best_model_path = f"models/best-model-{datetime.now():%Y%m%d-%H%M%S}.pth"



data_filter_config = {
    "acceleration_range": (-6, 6), # originally (-6, 6)
    "speed_range": (0, 30),
    "allow_leader_lc": False,
    "spacing_range": (10, 100), # used to rule out the pairs without CF.
    "length_thres": 6,
    "dtw_range": (0, 30),
    "thw": (0, 2),
    "pos_tol_range": (0, 2.5),
    "r_time_range": (0, 2),
    "spd_tol_range": (0, 100)
}

filter_names = [
    "time_headway_check",
    "all_same_leader",
    "speed_in_range",
    "veh_exist",
    # "dtw_in_range",
    "space_in_range",
    "inconsistent",
    # "reaction_in_range",
]

##### MODEL CONFIGS ######

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
    "loss_func": StyleLoss(style_data_config["y_groups"]["y_seq"]["features"]).acc_spacing_mse,
    "optim": optim.Adam,
    "lr": 5e-4, # transfollower is 1e-4
    "logger": {
        "mode": "normal"
    },
}


lstm_data_config = {
    "batch_size": 64,
    "train_data_ratio": 0.8,
    "seq_len": 40,
    "pred_len": 20,
    "stride": 20,
    "scaler": StandardScaler,
    "dataset": LSTMDataset,
    "in_features": [CF.SELF_V, CF.DELTA_X, CF.DELTA_V, CF.SELF_L, CF.LEAD_L],

    "eva_features": [CF.SELF_X, CF.SELF_V, CF.SELF_A, CF.DELTA_X, CF.LEAD_X]
}

lstm_model_config = {
    "model_name": "CF_LSTM",
    "num_feature": len(lstm_data_config["in_features"]),
    "pred_step": lstm_data_config["pred_len"],
    "batch_norm": False,
    
    "bidirectional": False,
    "dropout": 0,
    "regular_output": False,
    "regular_range": (-6, 6),  # disabled when regular_output set to false
    "regular_func": "tanh", # choose from tahn and sigmoid, disabled when regular_output set to false
    "activation_func": nn.Sigmoid(),
    "activate_name": "sigmoid",
    "state_num_feat": 10,
}

idm_calibration_config = {

    "x_groups": {"x": {"features": [CF.SELF_V, CF.LEAD_V, CF.DELTA_X]}},
    "y_groups": {"y": {"features": [CF.SELF_X, CF.SELF_V, CF.SELF_A]}},
    "downsample": 5, # step, sample every 5 step (0.5 second)
    "loss": IDMLoss().acc_dis_mse,
    "resolution": 0.1, # second
    "pred_horizon": 5, # step
    "historic_step": 5, # step
    "update_func": idm_update_train_series,
    "concat": idm_concat,
    "start_step": 5, # step
    "scaler": DataScaler(),
    "pred_func": DEFAULT_PRED_FUNC,
    "mask": DEFAULT_MASK,
    "randomseed": 42,
    "save_path": "./data/idm_calibration",
    "device": "cpu"
}



##### TEST CONFIGS ######


test_config = {
    "datapath": "F:\DATA\ZenTraffic\ZenTraffic90kalman_new.npy",
    "device": "cpu",
    "idm_state": [24.68, 1.67, 1.37, 1.68, 2.46],
    "lstm_state_path": ...,
    "style_state_path": ...,
    "transformer_state_path": ...,

    # style agent
    "style_agent":  
    { 
        "pred_func": DEFAULT_PRED_FUNC,
        "update_func": style_update_func,
        "mask": stylecf_mask,
        "concat_func": stack_name
    },


    # lstm agent
    "lstm_agent":  
    { 
        "pred_func": style_pred_func,
        "update_func": style_update_func,
        "mask": transformer_mask,
        "concat_func": stack_name
    },

    # transformer agent
    "transformer_agent":  
    { 
        "pred_func": style_pred_func,
        "update_func": style_update_func,
        "mask": transformer_mask,
        "concat_func": stack_name
    },

    # idm agent 
    "idm_agent":  
    { 
        "pred_func": DEFAULT_PRED_FUNC,
        "update_func": idm_update_func,
        "concat_func": idm_concat,
        "mask": DEFAULT_MASK,
    },
}
