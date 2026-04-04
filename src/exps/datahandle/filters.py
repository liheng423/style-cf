import numpy as np
from typing import Callable, List
from ..utils.datapack import SampleDataPack
from ...schema import CFNAMES as CF


def mask_min_self_id_samples(datapack: SampleDataPack, min_samples: int) -> np.ndarray:
    """
    Keep only samples whose SELF_ID has at least `min_samples` trajectories.
    """
    if min_samples <= 1:
        return np.ones(datapack.data.shape[0], dtype=bool)

    if CF.SELF_ID not in datapack.names:
        raise KeyError(f"{CF.SELF_ID} is required for min-self-id filtering")

    self_ids = datapack[:, 0, CF.SELF_ID].astype(np.int64)
    unique_ids, counts = np.unique(self_ids, return_counts=True)
    keep_ids = unique_ids[counts >= int(min_samples)]
    if keep_ids.size == 0:
        raise ValueError(
            f"No SELF_ID satisfies min_samples={min_samples}. "
            "Lower data_filter_config['min_self_id_samples']."
        )
    return np.isin(self_ids, keep_ids)


def filter_min_self_id_samples(datapack: SampleDataPack, min_samples: int) -> SampleDataPack:
    mask = mask_min_self_id_samples(datapack, min_samples=min_samples)
    return SampleDataPack(
        datapack.data[mask],
        datapack.names.copy(),
        datapack.rise,
        datapack.kph,
        datapack.kilo_norm,
        datapack.dt,
    )

class CFFilter:
    def __init__(self, datapack: SampleDataPack, filter_config):
        self.datapack = datapack
        self.data = datapack.data
        self.names = datapack.names
        self.config = filter_config

    def space_in_range(self) -> np.ndarray:
        delta_x = self.data[:, :, self.names[CF.DELTA_X]]
        lower, upper = self.config["spacing_range"]
        return np.all((delta_x >= lower) & (delta_x <= upper), axis=1)

    def veh_exist(self) -> np.ndarray:
        id_self = self.data[:, :, self.names[CF.SELF_ID]]
        id_lead = self.data[:, :, self.names[CF.LEAD_ID]]
        return np.all(id_self != -1, axis=1) & np.all(id_lead != -1, axis=1)
    
    def dtw_in_range(self) -> np.ndarray:
        from tslearn.metrics import dtw

        values = []
        for idx in range(self.data.shape[0]):
            v_self = self.data[idx, :, self.names[CF.SELF_V]]
            v_lead = self.data[idx, :, self.names[CF.LEAD_V]]
            dist = dtw(v_self, v_lead)
            values.append(dist)

        values = np.array(values)
        lower, upper = self.config["dtw_range"]
        return (values >= lower) & (values <= upper)
    
    def reaction_in_range(self) -> np.ndarray:
        from tslearn.metrics import dtw_path

        values = []
        for idx in range(self.data.shape[0]):
            v_self = self.data[idx, :, self.names[CF.SELF_V]]
            v_lead = self.data[idx, :, self.names[CF.LEAD_V]]
            path, _ = dtw_path(v_self, v_lead)
            time_delays = [abs(i - j) * 0.1 for i, j in path if i < len(v_self) and j < len(v_lead)]
            reaction_time = np.mean(time_delays)
            values.append(reaction_time)

        values = np.array(values)
        lower, upper = self.config["r_time_range"]
        return (values >= lower) & (values <= upper)

    def no_lane(self) -> np.ndarray:
        lc = self.data[:, :, self.names[CF.LC]]
        return np.all((lc != -1) & (lc != 1), axis=1)
    
    def speed_in_range(self) -> np.ndarray:
        v_self = self.data[:, :, self.names[CF.SELF_V]]
        v_lead = self.data[:, :, self.names[CF.LEAD_V]]
        lower, upper = self.config["speed_range"]

  
        in_range = (np.minimum(v_self, v_lead) >= lower) & (np.maximum(v_self, v_lead) <= upper)
     
        return np.all(in_range, axis=1)

    def acc_in_range(self) -> np.ndarray:
        acc_self = self.data[:, :, self.names[CF.SELF_A]]
        acc_lead = self.data[:, :, self.names[CF.LEAD_A]]
        lower, upper = self.config["acceleration_range"]

        in_range = (np.minimum(acc_self, acc_lead) >= lower) & (np.maximum(acc_self, acc_lead) <= upper)
        
        return np.all(in_range, axis=1)

    def all_same_leader(self) -> np.ndarray:
        assert CF.LEAD_ID in self.names.keys()
        ids = self.data[:, :, self.names[CF.LEAD_ID]]
        return np.all(ids == ids[:, [0]], axis=1)

    def all_same_self(self) -> np.ndarray:
        assert CF.SELF_ID in self.names.keys()
        ids = self.data[:, :, self.names[CF.SELF_ID]]
        return np.all(ids == ids[:, [0]], axis=1)

    def time_headway_check(self) -> np.ndarray:
        assert self.config["thw"][1] >= self.config["thw"][0]
        delta_x = self.data[:, :, self.names[CF.DELTA_X]] - self.data[:, :, self.names[CF.LEAD_L]]
        v_self = self.data[:, :, self.names[CF.SELF_V]]
        thw = delta_x / np.maximum(v_self, 1)
        thw_mean = np.mean(thw, axis=1)
        return (thw_mean < self.config["thw"][1]) & (thw_mean > self.config["thw"][0])

    def no_truck_self(self) -> np.ndarray:
        length = self.data[:, :, self.names[CF.SELF_L]]
        return np.all(length < self.config["length_thres"], axis=1)

    def no_truck_leader(self) -> np.ndarray:
        length = self.data[:, :, self.names[CF.LEAD_L]]
        filter_index = np.all(length < self.config["length_thres"], axis=1)
    
        return filter_index
    
    def inconsistent(self) -> np.ndarray: 
        self_pos_cons, self_spd_cons = self.datapack.check_consistency()
        leader_pos_cons, leader_spd_cons = self.datapack.check_consistency(
            x_key=CF.LEAD_X,
            v_key=CF.LEAD_V,
            a_key=CF.LEAD_A,
        )

        self_pos_cons, self_spd_cons = np.abs(self_pos_cons).mean(axis=1), np.abs(self_spd_cons).mean(axis=1)
        leader_pos_cons, leader_spd_cons = np.abs(leader_pos_cons).mean(axis=1), np.abs(leader_spd_cons).mean(axis=1)

        pos_cons = np.maximum(self_pos_cons, leader_pos_cons)
        spd_cons = np.maximum(self_spd_cons, leader_spd_cons)

        pos_lower, pos_upper = self.config["pos_tol_range"]
        spd_lower, spd_upper = self.config["spd_tol_range"]

        filter_index = (pos_lower <= pos_cons) & (pos_cons <= pos_upper) & (spd_lower <= spd_cons) & (spd_cons <= spd_upper)

        return filter_index

    def filter(self, funcs: List[Callable]) -> SampleDataPack:
        masks = [func() for func in funcs]
        return SampleDataPack(self.datapack.data[np.logical_and.reduce(masks)], self.datapack.names, self.datapack.rise, self.datapack.kph, self.datapack.kilo_norm, self.datapack.dt)
