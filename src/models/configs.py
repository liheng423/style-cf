from sklearn.preprocessing import StandardScaler
from src.models.dataset import StyledTransfollowerDataset
from src.schema import CFNAMES as CF

stylecf_data_config = {
    "batch_size": 64,
    "train_data_ratio": 0.7,
    "seq_len": 60,# steps not seconds
    "label_len": 40,# steps not seconds
    "pred_len": 40, # steps not seconds
    "stride": 20, # steps not seconds
    "scaler": StandardScaler,
    "dataset": StyledTransfollowerDataset,
    
    "x_groups": [
        {"key": "enc_x", "features": [CF.SELF_V, CF.DELTA_X, CF.DELTA_V, CF.SELF_L, CF.LEAD_L], "transform": True},
        {"key": "dec_x", "features": [CF.SELF_V, CF.LEAD_V], "transform": True},
        {"key": "style", "features": [CF.REACT, CF.THW, CF.DELTA_X, CF.SELF_V], "transform": True},
    ],
    "y_groups": [
        {"key": "y_seq", "features": [CF.SELF_X, CF.SELF_V, CF.SELF_A, CF.DELTA_X, CF.LEAD_X]},
    ],
}
