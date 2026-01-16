import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from dataprocess.processor import *
from schema import *
from dataprocess.tableutils import *
from tqdm import tqdm
from collections import namedtuple
from typing import Any, Dict, Callable
import random
from dataprocess.filters import VehicleTimeFilter, filter_data
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
        # Filter out Col.ID and Col.TIME from extras to avoid replication
        self.extras = {k: v for k, v in extras.items() if k not in [Col.ID, Col.TIME]}
    
    def __eq__(self, other):
        if isinstance(other, VehicleTime) or other is None:
            return True if self.id == other.id and self.time == other.time else False
        return False
    
    def __bool__(self):
        """Returns False if dummy_vehicletime, otherwise True."""
        return self is not None and self.id != -1
    
    def __getitem__(self, key: str):
        if key in self.extras:
            return self.extras[key]

        if hasattr(self, key.lower()):
            return getattr(self, key.lower())
        
        raise KeyError(f"VehicleTime has no field '{key}'")
    
    def to_list(self):
        return [self.id, self.time] + list(self.extras.values())
    
    def names_list(self):
        return [Col.ID, Col.TIME] + list(self.extras.keys())
    
    



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
        # VehicleTimes are not tracked, to ensure the dataformat is consistent
        self.dummy_vt = VehicleTime(-1, -1, **{col: -1 for col in col_to_extract if col not in [Col.ID, Col.TIME]})

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
            for lane in self.layout["mainstream"]:
                lane_df = lookup(self.dataframe, None, time).loc[lambda df: df[Col.LANE] == lane]
                if self.rise:
                    self.veh_sort_graph[(time, lane)] = lane_df.sort_values(Col.KILO)
                else: 
                    self.veh_sort_graph[(time, lane)] = lane_df.sort_values(Col.KILO, ascending=False)

    def _get_veh(self, id, time) -> VehicleTime:
        
        assert isinstance(id, (int, np.number))
        assert isinstance(time, (float, int))
            
        datapoint = lookup(self.dataframe, id, time)
        
        assert datapoint is not None, "This key doesn't exist in the data"
        # assert datapoint[Col.SPD] >= 0, str(datapoint[Col.SPD])

        # Filter out index names from columns to extract to prevent KeyError
        cols_to_get = [c for c in self.col_to_extract if c not in self.dataframe.index.names]
        extras_dict = {col_name: datapoint[col_name] for col_name in cols_to_get}

        return VehicleTime(datapoint.name[0], datapoint.name[1], **extras_dict)

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
                cols_to_get = [c for c in self.col_to_extract if c not in self.dataframe.index.names]
                extras_dict = {col_name: front_vehicle[col_name] for col_name in cols_to_get}
                vehicles.append(VehicleTime(front_vehicle.name[0], front_vehicle.name[1], **extras_dict))
            if len(front_vehicles) == n:
                return vehicles
            else:
                assert len(front_vehicles) < n
                num_padding = n - len(vehicles)
                return vehicles + [self.dummy_vt for _ in range(num_padding)]
        
        else:
            assert front_vehicles.empty
            return [self.dummy_vt  for _ in range(n)]
    
    def _find_rear_vehicle(self, kilo, now_time, lane) -> VehicleTime:
        return self._find_front_vehicle(kilo, now_time, lane, not self.rise)

    def find_leader(self, veh: VehicleTime) -> VehicleTime:
        return self._find_front_vehicle(veh[Col.KILO], veh[Col.TIME], veh[Col.LANE])[0]
    
    def find_follower(self, veh: VehicleTime) -> VehicleTime:
        return self._find_rear_vehicle(veh[Col.KILO], veh[Col.TIME], veh[Col.LANE])[0]
    
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
        
        
        left_front = self._find_front_vehicle(veh["kilo"], veh["time"], left_lane)[0]  if left_lane else self.dummy_vt 
        left_rear = self._find_rear_vehicle(veh["kilo"], veh["time"], left_lane)[0]  if left_lane else self.dummy_vt 
    
        right_front = self._find_front_vehicle(veh["kilo"], veh["time"], right_lane)[0] if right_lane else self.dummy_vt 
        right_rear = self._find_rear_vehicle(veh["kilo"], veh["time"], right_lane)[0]  if right_lane else self.dummy_vt 

        return Neighbors(left_front=left_front, left_rear=left_rear, right_front=right_front, right_rear=right_rear)
        
    def find_n_leader(self, veh: VehicleTime, n: int) -> List[VehicleTime]:

        return self._find_front_vehicle(veh["kilo"], veh["time"], veh["lane"], self.rise, n=n)
    
    def _find_start_end_time(self, id) -> Tuple[float]:
        table = lookup(self.dataframe, id)
        time = table.index.get_level_values(Col.TIME)
        return (time.min(), time.max())
    
# %% ================ Trajectory Generation ====================


class SequentialIDGenerator:

    def __init__(self, start_id=0):
        self._current_id = start_id

    def generate_id(self):
        new_id = self._current_id
        self._current_id += 1
        return new_id
    
class SeriesIDGenerator:
    def __init__(self, id_list: List[int]):
        self.id_list = id_list
        self.index = 0

    def generate_id(self):
        if self.index < len(self.id_list):
            new_id = self.id_list[self.index]
            self.index += 1
            return new_id
        else:
            raise StopIteration("No more IDs available in the list.")
        



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
        
    def roll_and_filter(self, max_length: int, func: Callable) -> None:
        """
        Rolls through the trajectories, applies filters, and extracts segments.
        """
        assert isinstance(func, Callable)
        seg_length = self.window_len
        max_length = max_length
        
        i = 0
        while i + seg_length <= max_length: # Use <= to include the last possible segment
            flag, next_idx = func(i, seg_length)
            if not flag: i+= next_idx; continue
            i += self.jump()
    
    def roll_and_filter_with_leftover(self, max_length: int, func: Callable):
        pass



class TrajectoryExtractor:
    """
    This class is responsible for extracting trajectories of vehicles over specified durations.
    """
    def __init__(self, include_config: dict, window_roller: WindowRoller):
        assert "include_leader" in include_config, "No input field found in the config"
        assert "include_self" in include_config, "No input field found in the config"
        self.config = include_config
        self.window_roller = window_roller

    def find_trajectory(self, ex: VehicleTimeExtractor, id: int, find_function: Callable= lambda x: x, *func_args) -> np.ndarray:
        """
        Extract the trajectory of a vehicle by applying a specified function at each time step.
        """

        start, end = ex._find_start_end_time(id)
        
        # Determine the number of time steps
        num_steps = int(round((end - start) / ex.process_result.resolution)) + 1

        # Get column names for structured array dtype
        col_names = self.get_col_names(ex)
        traj = np.empty((num_steps, len(col_names)))

        for t_idx, t in enumerate(decimal_arange(start, end, ex.process_result.resolution)): 
            traj[t_idx] = find_function(ex._get_veh(id, t), *func_args).to_list()

        return traj
    
    def get_col_names(self, ex: VehicleTimeExtractor):

        # Get an arbitrary ID from the dataframe to create an example VehicleTime object
        # This assumes the dataframe is not empty.

        example_id = ex.dataframe.index.get_level_values(Col.ID).unique()[0]
        start_time, _ = ex._find_start_end_time(example_id)
        
        return ex._get_veh(example_id, start_time).names_list()

    def retrieve_by_id(self, ex: VehicleTimeExtractor, id: int, filters: Dict[str, List[Callable]]) -> Dict[str, np.ndarray]:
        
        include_leader = self.config["include_leader"]
        include_self = self.config["include_self"]

        # Prepare trajectories based on include_config
        trajectories = {}
        if include_self:
            trajectories["self"] = self.find_trajectory(ex, id)
        if include_leader:
            trajectories["leader"] = self.find_trajectory(ex, id, ex.find_leader)

        output_data = {} # to store the output np.ndarray segments
        max_length = trajectories["self"].shape[0]

        def handle_vehtime(i, seg_length):
            segment_passed_filters = True
            segments = {}
            next_idx = 0  # Initialize to prevent UnboundLocalError

            for role, traj in trajectories.items():
                segment = traj[i : i + seg_length]
                segments[role] = segment
                role_filters = filters.get(f"{role}", [])
                

                if_filtered, jump_distance = filter_data(segment, role_filters)
                if if_filtered:
                    # If a filter fails, we need to adjust the outer loop's 'i'
                    # to jump past the violating segment.
                    next_idx = jump_distance
                    segment_passed_filters = False
                    break # No need to check other roles if one fails

            if segment_passed_filters:
                for role, segment in segments.items():
                    output_data.setdefault(role, []).append(segment)
            return segment_passed_filters, next_idx

        # Apply window rolling and filtering
        self.window_roller.roll_and_filter(max_length, handle_vehtime)

        # Concatenate data for each role within this ID
        concat_data = {}
        for role, segments in output_data.items():
            if segments: # Only concatenate if there are segments
                concat_data[role] = np.concatenate(segments, axis=0)

        return concat_data
    
    def retrieve_all(self, ex: VehicleTimeExtractor, filters: dict[str, list[Callable]], id_generator: SequentialIDGenerator = None) -> Dict[str, np.ndarray]:
        """
        Extract trajectories for all vehicle IDs in the provided list.
        """
        all_data = {}
        id_list = ex.dataframe.index.get_level_values(Col.ID).unique()

        if id_generator is None: # use original IDs in the dataset
            id_generator = SeriesIDGenerator(id_list)

        for vid in tqdm(id_list):
            id_data = self.retrieve_by_id(ex, vid, filters)
            for role, data in id_data.items():
                all_data.setdefault(role, []).append(data)

        # Concatenate data for each role
        concatenated_data = {role: np.concatenate(data_list, axis=0) for role, data_list in all_data.items()}

        return concatenated_data