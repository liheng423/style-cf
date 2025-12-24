import numpy as np
from schema import *

class VehicleFilter:

    @staticmethod
    def filter_data(data: np.ndarray, point_filters, tra_filters) -> tuple[bool, int]:
        """
        Return if pass
        """
        next_indices = []
        
        # Check trajectory filters
        for f in tra_filters:
            passed, next_idx = f(data)
            if not passed:
                next_indices.append(next_idx)
        
        # Check point filters
        for pf in point_filters:
            if isinstance(pf, tuple):
                passed, next_idx = pf[0](data, pf[1])
            else:
                passed, next_idx = pf(data)
            if not passed:
                next_indices.append(next_idx)
        
        if next_indices:
            return (True, max(next_indices))
        else:
            return (False, 0)
        
    @staticmethod
    def _veh_exist(data: np.ndarray) -> tuple[bool, int]:
        assert isinstance(data, np.ndarray), "Only accept numpy array"
        if any(data[Col.ID] == -1): # if dummy vehicle
            violating_indices = np.where(data[Col.ID] == -1)[0]
            max_idx = violating_indices[-1]
            return (False, max_idx + 1)  # 下次从该索引后开始检查
        else:
            return (True, 0)
    

    @staticmethod
    def _veh_not_on_lane(data: np.ndarray, lane: int) -> tuple[bool, int]:
        assert isinstance(data, np.ndarray), "Only accept numpy array"
        lane_mask = data[Col.LANE] == lane
        if np.any(lane_mask):
            # 找到车辆在目标车道上的最大索引
            violating_indices = np.where(lane_mask)[0]
            max_idx = violating_indices[-1]
            return (False, max_idx + 1)  # 下次从该索引后开始检查
        else:
            return (True, 0)  # 所有时间点都不在目标车道

    @staticmethod
    def in_acc_range(data: np.ndarray, acc_range: Tuple) -> tuple[bool, int]:
        assert isinstance(data, np.ndarray), "Only accept numpy array"
        acc = data[Col.ACC]
        valid_mask = (acc >= acc_range[0]) & (acc < acc_range[1])
        if np.all(valid_mask):
            return (True, 0)  # 所有时间点加速度均在范围内
        else:
            # 找到加速度超出范围的最大索引
            violating_indices = np.where(~valid_mask)[0]
            max_idx = violating_indices[-1]
            return (False, max_idx + 1)
        
    @staticmethod
    def no_lc(data: np.ndarray) -> tuple[bool, int]:
        assert isinstance(data, np.ndarray), "Only accept numpy array"
        in_lane = data[Col.IN_LC]  # 获取车辆是否正在换道的信息
        if np.all(in_lane == 0):  # 如果所有时间点都不在换道状态
            return (True, 0)  # 无换道行为
        else:
            # 找到最后一次换道的索引
            change_indices = np.where(in_lane != 0)[0]
            if change_indices.size == 0:
                return (True, 0)  # 理论上不可能，但保留以防万一
            last_change_idx = change_indices[-1]
            return (False, last_change_idx + 1)  # 跳过换道点之后的数据