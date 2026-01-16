import datetime
from utils.utils import *
from typing import Union, Dict, List, NamedTuple, Tuple
from src.dataprocess.kalman import kf
import numpy as np
from schema import *
import pandas as pd
from src.dataprocess.tableutils import *


class ProcessResult(NamedTuple):
    data: pd.DataFrame
    rise: bool
    in_kph: bool
    resolution: float

class DataProcessor:

    def __init__(self, rise: bool, in_kph: bool, time_resolution: float):
        """
        Initialize the DataProcessor with a configuration dictionary.

        Args:
            rise (bool): Whether the KILO column represents rising positions.
            in_kph (bool): Whether the speed is in kilometers per hour.
            time_resolution (float): the time resolution in seconds.
        """ 
        self.in_kph = in_kph #  whether the speed is in kilometers per hour
        self.rise = rise # whether the KILO is rising or falling
        self.time_in_sec = False # whether the TIME column is in seconds
        self.time_resolution = time_resolution

    def get_result(self, dataframe: pd.DataFrame) -> ProcessResult:
        return ProcessResult(dataframe, self.rise, self.in_kph, self.time_resolution)

    
    def fix_rise(self, dataframe: pd.DataFrame):
        """
        Convert the KILO column to represent rising positions.
        """
        assert not self.rise, "Data is already in rising format"

        dataframe = dataframe.copy()
        dataframe[Col.KILO] = dataframe[Col.KILO].max() - dataframe[Col.KILO]
        self.rise = True
        return dataframe

    def to_kph(self, dataframe: pd.DataFrame):
        """
        Convert the speed column to kilometers per hour.
        """

        assert not self.in_kph, "Speed is already in kph"
        dataframe = dataframe.copy()
        dataframe[Col.SPD] = ms2kph(dataframe[Col.SPD])
        self.in_kph = True
        return dataframe

    def to_ms(self, dataframe: pd.DataFrame):
        """
        Convert the speed column to meters per second.
        """
        assert self.in_kph, "Speed is not in kph"
        dataframe = dataframe.copy()        
        dataframe[Col.SPD] = kph2ms(dataframe[Col.SPD])
        self.in_kph = False
        return dataframe

    def set_index(self, dataframe: pd.DataFrame):
        """
        Set the DataFrame index to a MultiIndex based on ID and TIME columns.
        This serves as the standard format for further data processing.

        Args:
            dataframe (pd.DataFrame): Input DataFrame.
        """
        dataframe = dataframe.copy()
 
        return dataframe.set_index([Col.ID, Col.TIME])

 

    def strtime2sec(self, dataframe: pd.DataFrame, time_format: str, start_time: datetime = None) -> pd.DataFrame:
        """
        Convert string time to seconds since a start time.

        Args:
            dataframe (pd.DataFrame): Input DataFrame containing time data.
            time_format (str): Format of the time strings.
            start_time (datetime, optional): Start time for conversion. Defaults to the minimum time in the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with time converted to seconds.
        """
        dataframe = dataframe.copy()
        start_time = start_time if start_time != None else strtime2datetime(dataframe[Col.TIME].min(), time_format)
        dataframe[Col.TIME] = batch_strtime2sec(dataframe[Col.TIME], start_time, time_format)
        
        self.time_in_sec = True # mark that TIME is now in seconds

        return dataframe

    def kalman_filter(self, dataframe: pd.DataFrame, kalman_params=None) -> pd.DataFrame:
        """
        Apply a Kalman filter to the DataFrame.

        Args:
            dataframe (pd.DataFrame): Input DataFrame.
            rise (bool): Direction of movement in KILO (rising or falling).
            in_kph (bool): Whether the speed is in kilometers per hour.
            kalman_params (dict, optional): Parameters for the Kalman filter.

        Returns:
            pd.DataFrame: DataFrame after applying the Kalman filter.
        """
        assert self.rise == True and self.in_kph == False, "Currently only support rise=True and in_kph=False"
        dataframe = dataframe.copy()

        # Define Kalman filter parameters
        # These parameters might need to be tuned based on the specific data characteristics
        kalman_params = {
            'uncertainty_init': 100,  # Initial uncertainty for position and velocity
            'uncertainty_pos': 0.1,   # Measurement noise for position
            'uncertainty_speed': 0.5, # Measurement noise for speed
            'max_acc': 5              # Maximum possible acceleration (m/s^2)
        } if kalman_params is None else kalman_params

        def apply_kf_to_group(group_df):
            # Prepare data for the kf function: [x, v, t]
            # Assuming 'KILO' is y (longitudinal position), 'SPD' is v, 'TIME' is t
            # x can be a dummy column or lateral position if available
            # For 1D motion, x is not used by the kf function itself, but the input format expects it.
            # Here, we'll assume 1D motion along the 'KILO' axis.
            veh_data = group_df[[Col.KILO, Col.SPD, Col.TIME]].copy()
            veh_data.insert(0, 'dummy_x', 0) # Add a dummy x column
            
            kf_results = kf(veh_data.values, kalman_params)
            group_df = group_df.copy()
            group_df[Col.KILO] = kf_results[:, 0]
            group_df[Col.SPD] = kf_results[:, 1]
            return group_df
        
        return dataframe.groupby(Col.ID, group_keys=False).apply(apply_kf_to_group).reset_index(drop=True)

    def check_consistency(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Check the consistency of the data by comparing position and speed.

        Args:
            dataframe (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with consistency checks.
        """
        dataframe = dataframe.copy()
        dt = Col.resolution
        incon_table = dataframe.groupby(Col.ID).apply(
            lambda df: ((df[Col.KILO].diff().fillna(0).abs() - kph2ms(df[Col.SPD]) * dt).abs() < 0.5).iloc[1:].mean()
        )
        return incon_table

    def generate_lc(self, dataframe: pd.DataFrame, window: float):
        """
        Include lane change (IN_LC) flags in the DataFrame.

        Args:
            dataframe (pd.DataFrame): Input DataFrame.
            window (float): Time window for lane-changing flagging, indicating the duration of lane-changing.

        Returns:
            pd.DataFrame: DataFrame with LC flags (if the vehicle is in LC).
        """
        dataframe = dataframe.copy()
        window = int(window)
        lc_series = pd.Series(0, index=dataframe.index)

        for vehicle_id, group in dataframe.groupby(Col.ID):
            lc_indices = find_jumps(group[Col.LANE]).index

            for idx in lc_indices:
                time_values = group[Col.TIME]
                assert time_values.is_monotonic_increasing, f"TIME index is not sorted for vehicle {vehicle_id}"
                pos = group.index.get_loc(idx)
                start = max(0, pos - window // 2)
                mid = start + window // 2
                end = min(len(group) - 1, start + window)
                lc_series.loc[group.index[start:mid]] = -1
                lc_series.loc[group.index[mid:end + 1]] = 1

        dataframe[Col.LC] = lc_series
        return dataframe

    def derive_acc(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Derive acceleration (ACC) based on speed and time.

        Args:
            dataframe (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with derived accelerations.
        """
        dataframe = dataframe.copy()
        grouped = dataframe.groupby(Col.ID)
        in_kph = self.in_kph

        def compute_acceleration(group):
            time_values = group[Col.TIME].to_numpy()
            if in_kph:
                speed_values = kph2ms(group[Col.SPD].to_numpy())
            else:
                speed_values = group[Col.SPD].to_numpy()
            acc_values = np.diff(speed_values) / np.diff(time_values)
            acc_values = np.insert(acc_values, 0, acc_values[0])
            group[Col.ACC] = acc_values
            return group

        dataframe = grouped.apply(compute_acceleration).reset_index(drop=True)
        return dataframe
