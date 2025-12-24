import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from processor import *
from schema import *
from tableutils import *
from tqdm import tqdm
from collections import namedtuple
from typing import Any, Dict
import random
from filters import VehicleFilter
# %% ================ Single DataPoint (VehicleTime) Extraction ====================
@dataclass(slots=True)
class VehicleTime:

    ### identifier ###
    id: int
    time: float


    ### user-defined fields ###
    extras: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, id, time, **extras):
        self.id = id
        self.time = time
        self.extras = extras
    
    def __eq__(self, other):
        if isinstance(other, VehicleTime) or other is None:
            return True if self.id == other.id and self.time == other.time else False
        return False
    
    def __bool__(self):
        """Returns False if dummy_vehicletime, otherwise True."""
        return self is not None and self != dummy_vehicletime
    
    def __getitem__(self, key: str):
        if key in self.extras:
            return self.extras[key]

        if hasattr(self, key):
            return getattr(self, key)
        
        raise KeyError(f"VehicleTime has no field '{key}'")
    
    def to_list(self):
        return [self.id, self.time] + list(self.extras.values())
    
    def names_list(self):
        return [Col.ID, Col.TIME] + list(self.extras.keys())
    
    

# VehicleTimes are not tracked, to ensure the dataformat is consistent
dummy_vehicletime = VehicleTime(-1, -1, None)

Neighbors = namedtuple("Neighbor", ["left_front", "left_rear", "right_front", "right_rear"])



class VehicleTimeExtractor:

    """
    This class is responsible for extracting `VehicleTime` objects and finding relationships between vehicles (leader, follower, neighbors) at specific points in time.
    It works with a processed DataFrame (`ProcessResult`) and a layout configuration
    to efficiently query vehicle data.
    The class aims to facilitate the tracking of car-following pairs and other
    inter-vehicle relationships.
    Note: the aim of this class is to identify the leader at a certain time for a given vehicle, since the lane-changing exists. 
    """


    def __init__(self, process_result: ProcessResult, layout: dict, col_to_extract: list[str]):
        self.process_result = process_result
        self.dataframe: pd.DataFrame = process_result.data
        self.layout = layout # to identify the neighbor lanes
        self.col_to_extract = col_to_extract
        self.veh_sort_graph = {} # for speeding-up the extraction
        self.rise = True

    def _validation(self):
        assert self.process_result.rise == True
        assert Col.ID in self.dataframe.columns
        assert Col.TIME in self.dataframe.columns


    ### UTILS ###

    def _find_start_end_time(self, id) -> Tuple[float]:
        table = lookup(self.dataframe, id)
        time = table.index.get_level_values(Col.TIME)
        return (time.min(), time.max())

    def _generate_veh_sort_graph(self) -> dict:
        """
        Generate mainstream lanes and time vehicle sort graph as cache, speed up the extraction
        """
        time_start = self.dataframe.index.get_level_values(Col.TIME).min()
        time_end = self.dataframe.index.get_level_values(Col.TIME).max()

        for time in tqdm(decimal_arange(time_start, time_end, self.process_result.resolution)):
            for lane in self.layout.mainstream:
                lane_df = lookup(self.dataframe, None, time).loc[lambda df: df[Col.LANE] == lane]
                if self.rise:
                    self.veh_sort_graph[(time, lane)] = lane_df.sort_values(Col.KILO)
                else: 
                    self.veh_sort_graph[(time, lane)] = lane_df.sort_values(Col.KILO, ascending=False)

    def _get_veh(self, id, time) -> VehicleTime:
        
        assert isinstance(id, int)
        assert isinstance(time, (float, int))
            
        datapoint = lookup(self.dataframe, id, time)
        
        assert datapoint is not None, "This key doesn't exist in the data"
        # assert datapoint[Col.SPD] >= 0, str(datapoint[Col.SPD])

        return VehicleTime(datapoint.name[0], datapoint.name[1], datapoint.name[1], {col_name: datapoint[col_name] for col_name in self.col_to_extract})

    def _query_leader(self, time, lane, kilo, n):
        """
        Retrieve the nearest n vehicles ahead or behind based on kilopost.
        """
        key = (time, lane)
        
        if key in self.veh_sort_graph:
            kilo_series = self.veh_sort_graph[key][Col.KILO].values
            
            # if rise, then veh_sort_graph is ascending,
            # if not rise, then veh_sort_graph is descending
            # the leader will always be in the following several index

            leader_idx = search_sorted(kilo_series, kilo) + 1
            return self.veh_sort_graph[key].iloc[leader_idx: min(leader_idx + n, len(kilo_series))]
        
        # If not cached, retrieve from the dataframe
        static_lane_df = lookup(self.dataframe, time_slice=time)
        static_lane_df = static_lane_df[static_lane_df[Col.LANE] == lane]
        
        if self.rise:
            return static_lane_df[static_lane_df[Col.KILO] > kilo].nsmallest(n, Col.KILO)
        else:
            return static_lane_df[static_lane_df[Col.KILO] < kilo].nlargest(n, Col.KILO)

    def _find_front_vehicle(self, kilo, now_time, lane, n=1) -> Union[List[VehicleTime], VehicleTime]:
        """
        Given location and time, find its leader at the same time
        """

        # Driving direction: increasing kilopost -> find the next larger kilopost
        front_vehicles: pd.DataFrame = self._query_leader(now_time, lane, kilo, n)
   

        if not front_vehicles.empty:

            vehicles = []
            for _, front_vehicle in front_vehicles.iterrows():
                vehicles.append(VehicleTime(front_vehicle.name[0], front_vehicle.name[1], {col_name: front_vehicle[col_name] for col_name in self.col_to_extract}, 
                                        front_vehicle[Col.LC]))
            if len(front_vehicles) == n:
                return vehicles
            else:
                assert len(front_vehicles) < n
                num_padding = n - len(vehicles)
                return vehicles + [dummy_vehicletime for _ in range(num_padding)]
        
        else:
            assert front_vehicles.empty
            return [dummy_vehicletime for _ in range(n)]
    
    def _find_rear_vehicle(self, kilo, now_time, lane) -> VehicleTime:
        return self._find_front_vehicle(kilo, now_time, lane, not self.rise)

    def find_leader(self, veh: VehicleTime) -> VehicleTime:
        return self._find_front_vehicle(veh["kilo"], veh["time"], veh.lane)[0]
    
    def find_follower(self, veh: VehicleTime) -> VehicleTime:
        return self._find_rear_vehicle(veh["kilo"], veh["time"], veh.lane)[0]
    
    def _get_adj_lanes(layout: dict, lane: int, kilo: float, rise, drive_right):
        """
        
        """

        assert layout["Range"][0] <= layout["Range"][1] if rise else layout["Range"][0] >= layout["Range"][1]
        for sec in layout["Sections"]:
            # If kilo is in the section range (accounting for rise),
            # we check lane adjacency within this section.
            if is_in_range(kilo, sec["range"], not rise):

                assert lane in sec["lanes"], "No such lane found in this section"

                lanes = sec["lanes"]
                sep = sec["seperation"]
                if sep == -1: # no seperation
                    group = lanes
                else: # lane seperation at given split point
                    group1, group2 = lanes[:sep], lanes[sep:]

                    # Determine which group the current lane belongs to.
                    group = group1 if lane in group1 else group2 if lane in group2 else None
                    if group is None:
                        raise ValueError("The lane is not found in any lane group")

                idx = group.index(lane)
                left_lane = right_lane = None

                if drive_right:
                    if idx - 1 >= 0: left_lane = group[idx - 1]
                    if idx + 1 < len(group): right_lane = group[idx + 1]
                else:
                    if idx + 1 < len(group): left_lane = group[idx + 1]
                    if idx - 1 >= 0: right_lane = group[idx - 1]

                return [left_lane, right_lane]

        raise ValueError("No section covers the given kilometer value.")
        
    def find_neighbors(self, veh) -> Neighbors[VehicleTime]:
        """
        Given id and time, find its neighbors (on the adjacent lanes)
        [0]: left_front vehicle
        [1]: left_rear vehicle
        [2]: right_front vehicle
        [3]: right_rear vehicle
        """

        left_lane, right_lane = self._get_adj_lanes(self.layout, veh["lane"], veh["kilo"], self.rise, self.layout["drive_right"])
        
        
        left_front = self._find_front_vehicle(veh["kilo"], veh["time"], left_lane)[0]  if left_lane else dummy_vehicletime
        left_rear = self._find_rear_vehicle(veh["kilo"], veh["time"], left_lane)[0]  if left_lane else dummy_vehicletime
    
        right_front = self._find_front_vehicle(veh["kilo"], veh["time"], right_lane)[0] if right_lane else dummy_vehicletime
        right_rear = self._find_rear_vehicle(veh["kilo"], veh["time"], right_lane)[0]  if right_lane else dummy_vehicletime

        return Neighbors(left_front=left_front, left_rear=left_rear, right_front=right_front, right_rear=right_rear)
        
    def find_n_leader(self, veh: VehicleTime, n: int) -> List[VehicleTime]:

        return self._find_front_vehicle(veh["kilo"], veh["time"], veh["lane"], self.rise, n=n)
    
    def _find_start_end_time(self, id) -> Tuple[float]:
        table = lookup(self.dataframe, id)
        time = table.index.get_level_values(Col.TIME)
        return (time.min(), time.max())
    
# %% ================ Trajectory Generation ====================

class id_counter:
    pass


class WindowRoller:

    def __init__(self, type: str, window_len, min_jump):
        self.type = type
        self.window_len = window_len
        self.min_jump = min_jump

    def _random_jump(self):
        
        return random.randint(self.min_jump + 1, self.window_len)
    
    def _full_jump(self):
        
        return self.window_len
    
    def jump(self):

        if self.type == "random_roll":
            return self._random_jump()
        elif self.type == "no_overlap":
            return self._full_jump()
        else:
            raise(RuntimeError, "Bad Config Input.")
        



class TrajectoryExtractor:
    """
    This class is responsible for extracting trajectories of vehicles over specified durations.
    """
    def __init__(self, include_config: dict, window_roller: WindowRoller):
        assert "include_neighbor" in include_config, "No input field found in the config"
        assert "include_leader" in include_config, "No input field found in the config"
        assert "include_self" in include_config, "No input field found in the config"
        self.config = include_config
        self.window_roller = window_roller

    def find_trajectory(self, ex: VehicleTimeExtractor, id: int, find_function: function= lambda x: x, *func_args) -> np.ndarray:
        """
        Extract the trajectory of a vehicle by applying a specified function at each time step.
        """

        start, end = ex._find_start_end_time(id)
        
        # Determine the number of time steps
        num_steps = int(round((end - start) / ex.process_result.resolution)) + 1

        # Get column names for structured array dtype
        col_names = self.get_col_names(ex)
        dtype = [(name, float) for name in col_names] # Assuming all values are float
        traj = np.empty(num_steps, dtype=dtype)

        for t_idx, t in enumerate(decimal_arange(start, end, ex.process_result.resolution)): # type: ignore
            traj[t_idx] = find_function(ex._get_veh(id, t), *func_args).to_list()

        return traj
    
    def get_col_names(self, ex: VehicleTimeExtractor):

        # Get an arbitrary ID from the dataframe to create an example VehicleTime object
        # This assumes the dataframe is not empty.

        example_id = ex.dataframe[Col.ID].iloc[0]
        start_time, _ = ex._find_start_end_time(example_id)

        return ex._get_veh(example_id, start_time).names_list()

    def retrieve(self, ex: VehicleTimeExtractor, duration: float, id: int, filters):
        
        include_neighbor = self.config["include_neighbor"]
        include_leader = self.config["include_leader"]
        include_self = self.config["include_self"]


        if include_neighbor:
            neighbor_traj = self.find_trajectory(id, ex.find_neighbors)
        if include_leader:
            leader_traj = self.find_trajectory(id, ex.find_leader)
        follower_traj = self.find_trajectory(id)



        ## window roller ##

        seg_length = int(duration / ex.process_result.resolution)  # Compute segment length
        max_length = follower_traj.shape[0]
        roller = self.window_roller

        #############################

        output_data = {} # to store the output np.ndarray segments
        
        

        i = 0
        while i + seg_length < max_length:

            segment_passed_filters = True
            segments = {}

            for role, traj, point_filters, traj_filters in [
                ("neighbor", neighbor_traj, filters["nei_point_filter_set"], filters["nei_traj_filter_set"]) if include_neighbor else (None, None, None, None),
                ("self", follower_traj, filters["self_point_filter_set"], filters["self_traj_filter_set"]) if include_self else (None, None, None, None),
                ("leader", leader_traj, filters["leader_point_filter_set"], filters["leader_traj_filter_set"]) if include_leader else (None, None, None, None),
            ]:
                if role:
                    segment = traj[i : i + seg_length, :]
                    segments[role] = segment
                    if_filtered, next_idx = VehicleFilter.filter_data(segment, point_filters, traj_filters) # check filters and provide next index (the index where error pops up) to jump to 
                    if if_filtered:
                        i += next_idx # jump to the next index that may pass the filters
                        segment_passed_filters = False
                        break # No need to check other roles if one fails
            if not segment_passed_filters: continue # if any segment fails, skip to the next iteration

            # Store the segments that passed the filters
            for role, segment in segments.items():
                output_data.setdefault(role, []).append(segment)

            i += roller.jump()  # 


        concat_data = {key: np.concatenate(np.expand_dims(value, 0), axis=0) for key, value in output_data.items()}

        return concat_data