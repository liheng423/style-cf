
import numpy as np
from tqdm import tqdm
from tslearn.metrics import dtw_path

def reaction_time(leader_v: np.ndarray, self_v: np.ndarray, time: np.ndarray):
    """
    Estimate the average reaction time of a follower vehicle relative to a leader vehicle
    based on the DTW alignment of their speed profiles. Also calculates the average delay
    for each time point.

    Parameters:
    - leader_v: ndarray of leader vehicle speeds (length T)
    - self_v: ndarray of follower vehicle speeds (length T)
    - time: ndarray of timestamps (length T)

    Returns:
    - avg_reaction_time: scalar, average reaction delay over all time points
    - avg_delay_per_timepoint: ndarray (length T), average delay at each time point
    - dtw_path: list of tuples, DTW matching path [(follower_idx, leader_idx), ...]
    """

    path, _ = dtw_path(self_v, leader_v)
    T = len(time)
    delay_dict = {t: [] for t in range(T)}

    for follower_idx, leader_idx in path:
        if 0 <= follower_idx < T and 0 <= leader_idx < T:
            delay = time[follower_idx] - time[leader_idx]
            delay = np.minimum(np.maximum(0.5, delay), 4)
            delay_dict[follower_idx].append(delay)

    avg_delay_per_timepoint = np.zeros(T)
    for t in range(T):
        if delay_dict[t]:
            avg_delay_per_timepoint[t] = np.mean(delay_dict[t])
        else:
            avg_delay_per_timepoint[t] = np.nan 


    return avg_delay_per_timepoint


def time_headway(spacing: np.ndarray, self_v: np.ndarray):
    """
    Compute Time Headway (THW) as spacing divided by ego vehicle speed.

    A small constant (1e-1) is added to the denominator to avoid division by zero.

    Parameters:
    - spacing: ndarray, distance between the leader and the ego vehicle (in meters)
    - self_v: ndarray, speed of the ego vehicle (in meters/second)

    Returns:
    - thw: ndarray, time headway values (in seconds)
    """
    thw = spacing / (self_v + 1e-1)

    return thw
 

def batch_apply(
    func,
    args_list
) -> np.ndarray:
    """
    Apply a single-sample function to batched inputs and return a stacked numpy array.

    Parameters:
    - func: a function that accepts N arguments (e.g., reaction_time or time_headway)
    - args_list: list of arrays, each with shape (N, ...)

    Returns:
    - results: np.ndarray with shape (N, ...) depending on func's output shape
    """
    num_samples = args_list[0].shape[0]
    results = [func(*(arg[i] for arg in args_list)) for i in tqdm(range(num_samples))]
    return np.stack(results) 
