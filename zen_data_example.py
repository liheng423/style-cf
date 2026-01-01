# %% Zen-traffic data processing example



import os
import numpy as np
import pandas as pd
from dataprocess.extractor import *
from dataprocess.tableutils import *
from utils.utils import *
from dataprocess.processor import *
import json
from dataprocess.filters import *

# %% Configuration for Zen-traffic data processing
DATA_FILE = "F:\DATA\Zen2\Zen-traffic\Data"
ROAD_SECTION_NAMES = {1: 11, 2: 4, 3: 13}
ZEN_TIMEFORMAT = "%H%M%S%f"
LAYOUT_FOLDER = ".\data\layouts"



data_file = lambda route, sec: os.path.join(DATA_FILE, f"HanshinExpresswayRoute{ROAD_SECTION_NAMES[route]}", "TrafficData", "TRAJECTORY", f"L00{route}_F00{sec}_trajectory.csv")


# %% ================ Data Column Names ===================

column_names = ["ID",  # vehicle ID
                "TIME",  # time (originally in str)
                "TYPE",  # vehicle type (1: normal vehicle, 2: large vehicle(bus, truck, etc.))
                "SPD", # speed (kph)
                "LANE", # lane number
                "LAT", # lateral position
                "LON", # longitudinal position
                "KILO", # position (meter)
                "LEN", # vehicle length (meter)
                "OBS"] # observation flag (1: detected record by image recognition, 0: interpolated record)

names_mapping = {
    "TIME": "TIME", 
    "ID": "ID",
    "SPD": "SPD",
    "KILO": "KILO",
    "LANE": "LANE",
    "LC": "LC",
    "ACC": "ACC"
}


# %% ================ Load Raw Data ===================

route = 1
scene = 1

# load raw data
raw_data = pd.read_csv(data_file(route, scene), names=column_names)

# load route layout
with open(os.path.join(LAYOUT_FOLDER, f"Route{int(ROAD_SECTION_NAMES[route])}.json"), 'r') as layout_file:
        layout = json.load(layout_file)

zen_processor = DataProcessor(rise=False, in_kph=True, time_resolution=0.1)

data = zen_processor.strtime2sec(raw_data, ZEN_TIMEFORMAT) # convert time from str to seconds
data = zen_processor.to_ms(data) # convert kph to m/s
data = zen_processor.fix_rise(data) # fix the KILO to be rising
data = zen_processor.kalman_filter(data)
data = zen_processor.set_index(data)

print(lookup(data, time_slice=(10, 20)))

process_result = zen_processor.get_result(data)


# %% ================ Extract CF pairs for training ===================

extract_config = {
    "Columns": ["ID", "TIME", "SPD", "KILO", "LANE", "LC", "ACC"]
}

def gen_ex(process_result: ProcessResult) -> VehicleTimeExtractor:
    ex = VehicleTimeExtractor(process_result, layout, extract_config["Columns"])
    ex._generate_veh_sort_graph() # to speed-up the extraction process, pre-cache the sorted vehicle platoon on each lane.
    return ex

ex = gen_ex(process_result)


# %% ================ Set up filters ===================

self_filter_set = [(VehicleTimeFilter.in_acc_range, (-20, 6)), (VehicleTimeFilter._veh_not_on_lane, 3), VehicleTimeFilter._veh_exist, VehicleTimeFilter.no_lc]

leader_filter_set = [(VehicleTimeFilter.in_acc_range, (-20, 6)), VehicleTimeFilter._veh_exist]


include_config = {
    "include_neighbor": True,
    "include_leader": True,
    "include_self": True
}

filters = {
    "self": self_filter_set,
    "leader": leader_filter_set,
}

seg_length = 300 # segment length in frames (30 seconds)

roller = WindowRoller("random_roll", window_size=seg_length, step_size= int(seg_length // 2))

traj_ex = TrajectoryExtractor(include_config, roller)


traj_ex.retrieve_all(ex, filters)
# %%
