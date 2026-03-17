
# %autoreload 2
# %% Imports

from src.exps.models.idm import IDM
from src.exps.configs import data_filter_config, filter_names, idm_calibration_config
from src.exps.idm_calibrate import calibrate_idm
from src.exps.train.model_trainer import build_dataset
from src.exps.utils.utils import build_id_datapack, load_zen_data


# %% 
def _dataset(head=None):

    data_path = "/Users/blow/datasets/DATA/ZenTraffic/ZenTraffic30.npy" 

    d = load_zen_data(data_path, rise=True, in_kph=False, kilo_norm=True)
    return d.head(head) if head is not None else d

def main(): 

    datapack = _dataset(10000)
  
    datapack = build_dataset(datapack, filter_names, data_filter_config)
    id_datapack = build_id_datapack(
        datapack, require_const_self_id=True, key_by_id=False
    )

    calibrate_idm(IDM, id_datapack, idm_calibration_config)


if __name__ == "__main__":
    main()
# %%
