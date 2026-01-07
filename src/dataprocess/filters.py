import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Callable
import numpy as np
from schema import *


def filter_data(data: np.ndarray, filters: list[Callable]) -> tuple[bool, int]:
        """
        Applies a list of filters to the data.
        """
        next_indices = []
        
        for f in filters:
            passed, next_idx = f(data) 
            
            if not passed:
                next_indices.append(next_idx)

        if next_indices:
            return (True, max(next_indices))  # Violation, return the maximum jump index

        return (False, 0)  # Passed

class VehicleTimeFilter:

    def __init__(self, name_list: list[str]):
        self.name2idx = {name: idx for idx, name in enumerate(name_list)}

    
    
        

    def _veh_exist(self,data: np.ndarray) -> tuple[bool, int]:
        assert isinstance(data, np.ndarray), "Only accepts numpy array"
        if np.any(data[self.name2idx[Col.ID]] == -1): # Check if any dummy vehicle (-1 ID) exists in the trajectory
            violating_indices = np.where(data[self.name2idx[Col.ID]] == -1)[0]
            max_idx = violating_indices[-1] # Get the last index where a dummy vehicle was found
            return (False, max_idx + 1) # Returns False (violation) and the index after the last dummy vehicle
        else:
            return (True, 0) # Returns True (no violation) and 0 (no jump needed)
    


    def _veh_not_on_lane(self, lane: int) -> tuple[bool, int]:

        def execute(data: np.ndarray):
            assert isinstance(data, np.ndarray), "Only accepts numpy array"
            lane_mask = data[self.name2idx[Col.LANE]] == lane # Create a boolean mask for vehicles on the specified lane
            if np.any(lane_mask):
                violating_indices = np.where(lane_mask)[0]
                max_idx = violating_indices[-1]
                return (False, max_idx + 1)  
            else:
                return (True, 0)
            
        return execute


    def in_acc_range(self, acc_range: tuple) -> tuple[bool, int]:
        def execute(data: np.ndarray):
            assert isinstance(data, np.ndarray), "Only accepts numpy array"
            acc = data[self.name2idx[Col.ACC]] # Extract acceleration data
            valid_mask = (acc >= acc_range[0]) & (acc < acc_range[1]) # Check if acceleration is within the specified range
            if np.all(valid_mask): # If all acceleration values are within range
                return (True, 0) # Return True (no violation) and 0 (no jump needed)
            else:
                # Find the maximum index where acceleration was out of range
                violating_indices = np.where(~valid_mask)[0] 
                max_idx = violating_indices[-1] # Get the last index where acceleration was out of range
                return (False, max_idx + 1) # Return False (violation) and the index after the last violation
        return execute
            

    def no_lc(self, data: np.ndarray) -> tuple[bool, int]: # No lane change
        assert isinstance(data, np.ndarray), "Only accepts numpy array"
        in_lane = data[self.name2idx[Col.LC]] # Get information about whether the vehicle is currently changing lanes
        if np.all(in_lane == 0): # If the vehicle is not in a lane change state at all time points
            return (True, 0) # Return True (no lane change) and 0 (no jump needed)
        else:
            # Find the index of the last lane change
            change_indices = np.where(in_lane != 0)[0] 
            if change_indices.size == 0: # This case should theoretically not happen if the 'else' block is entered.
                return (True, 0) # But included for robustness
            last_change_idx = change_indices[-1] # Get the last index where a lane change occurred
            return (False, last_change_idx + 1) # Return False (lane change occurred) and the index after the last lane change