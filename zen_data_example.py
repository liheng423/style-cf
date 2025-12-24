import os
import numpy as np
import pandas as pd
from data.data_utils import *
from utils.utils import *
from data.processor import *


# %% Configuration for Zen-traffic data processing
DATA_FILE = "F:\DATA\Zen2\Zen-traffic\Data"
ROAD_SECTION_NAMES = {1: 11, 2: 4, 3: 13}
ZEN_TIMEFORMAT = "%H%M%S%f"
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

raw_data = pd.read_csv(data_file(1, 1), names=column_names)

zen_processor = DataProcesser(names_mapping, rise=False, in_kph=True)

data = zen_processor.strtime2sec(raw_data, ZEN_TIMEFORMAT) # convert time from str to seconds
data = zen_processor.to_ms(data) # convert kph to m/s
data = zen_processor.fix_rise(data) # fix the KILO to be rising
data = zen_processor.kalman_filter(data)
data = zen_processor.set_index(data)
# print(lookup_df(raw_data, time_slice=(0, 100)))
print(lookup(data, time_slice=(10, 20)))


# %% ================ Extract CF pairs for training ===================

extract_config = {
    "Columns": ["ID", "TIME", "SPD", "KILO", "LANE", "LC", "ACC"]
}

def gen_ex(data):
    ex = Extracter(data, TIME, ID, LAYOUT, RISE, data_config)
    # to speed-up the extraction process, pre-cache the sorted vehicle platoon on each lane.
    # ex._generate_veh_sort_graph()

    return ex


ex = gen_ex(data)