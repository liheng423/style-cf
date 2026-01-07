import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import numpy as np
from schema import *
import bisect

kph2ms = lambda kph: kph / 3.6
ms2kph = lambda ms: ms * 3.6

def find_jumps(lane_sr: pd.Series) -> pd.Series:
    """
    Given only one vehicle_id trajectory, return at what time point it changes lane
    """
    # Compute diff on traffic_lane column; note that the first row will have NaN
    lane_change = lane_sr.diff().ne(0, fill_value=0)
    return lane_sr[lane_change]

def lookup(traj_df, id_slice=None, time_slice=None, distance_slice=None) -> pd.DataFrame:

    
    assert isinstance(traj_df.index, pd.MultiIndex) and len(traj_df.index.levels) == 2, "The DataFrame must have a MultiIndex with two levels (ID and TIME (in seconds))"

    assert Col.KILO in traj_df.columns, "KILO column not found in the DataFrame"

    slices = [id_slice, time_slice, distance_slice]

    for i in range(len(slices)):
        
        if slices[i] == None:
            slices[i] = slice(None)
            continue

        if isinstance(slices[i], (int, float, np.number)):
            if isinstance(slices[i], np.integer): # Use np.integer for broader integer type checking
                slices[i] = int(slices[i])
            continue

        if isinstance(slices[i], list):
            continue

        if not isinstance(slices[i], slice):
            assert (isinstance(slices[i], (tuple, set)) and len(slices[i]) == 2), "Not supported for this tuple-like input"
            assert slices[i][0] <= slices[i][1], "The latter must be larger than the former"

            slices[i] = slice(slices[i][0], slices[i][1])
            continue

    id_slice = slices[0]
    time_slice = slices[1]
    distance_slice = slices[2]

    assert not isinstance(distance_slice, list), "Distance slice can't be a list"
    
    if distance_slice.start != None and distance_slice.stop != None:
        return traj_df.loc[(id_slice, time_slice), : ].loc[(traj_df[Col.KILO] <= distance_slice.stop) & (traj_df[Col.KILO] >= distance_slice.start)]
    elif distance_slice.start == None and distance_slice.stop == None:
        return traj_df.loc[(id_slice, time_slice), : ]
    elif distance_slice.stop == None:
        return traj_df.loc[(id_slice, time_slice), : ].loc[traj_df[Col.KILO] >= distance_slice.start]
    else:
        return traj_df.loc[(id_slice, time_slice), : ].loc[traj_df[Col.KILO] <= distance_slice.stop]


def find_jumps(series: pd.Series, fill_value=0) -> pd.Series:
    """
    Given a pandas Series (e.g., representing a vehicle's trajectory for a specific attribute),
    return the time points where the value changes (jumps).

    Args:
        series (pd.Series): A pandas Series, typically representing an attribute over time for a single entity.
        fill_value (int, optional): The value to use for the first difference, as it will be NaN. Defaults to 0.

    Returns:
        pd.Series: A Series containing the values of the original series at the time points where a jump occurred.
    """
    jumps = series.diff().ne(0, fill_value=fill_value)
    return series[jumps]


def search_sorted(arr, x):
    """Search for x in a sorted array (ascending or descending) using bisect."""
    if arr.size == 0:  # Handle empty array
        return None  
    
    is_ascending = arr[0] < arr[-1]  # Detect sorting order
    
    if is_ascending:
        idx = bisect.bisect_left(arr, x)  # Binary search for ascending array
    else:
        # We perform a binary search on a reversed view and transform the index
        idx = bisect.bisect_left(arr[::-1], x)
        idx = len(arr) - 1 - idx  # Convert to the correct index in the original array
    
    return idx