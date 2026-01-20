
from torch import Tensor
import torch
import numpy as np

@staticmethod
def _predict_kinematics(accs: Tensor, initial_states: Tensor, dt: float) -> Tensor:
    """
    Compute predicted speed and distance based on acceleration.
    
    Args:
        accs (torch.Tensor): Acceleration tensor of shape (T,).
        dt (float): Time step duration.

    """
    init_dis, init_spd = initial_states[0], initial_states[1]
    accs = accs.flatten()
    pred_spd = init_spd + torch.cumsum(accs, dim=0) * dt
    pred_dis = init_dis + torch.cumsum(pred_spd * dt, dim=0)
    return torch.vstack([pred_dis, pred_spd, accs]).T


@staticmethod
def _predict_kinematics_np(accs: np.ndarray, initial_states: np.ndarray, dt: float) -> np.ndarray:
    """
    NumPy version of kinematics prediction (supports 1D or 2D batch).

    Args:
        accs (np.ndarray): Acceleration array of shape (T,) or (N, T).
        initial_states (np.ndarray): Initial [displacement, speed], shape (2,) or (N, 2).
        dt (float): Time step duration.
    Return:
        np.ndarray: Predicted kinematics of shape (T, 3) or (N, T, 3).
    """
    accs = np.asarray(accs)
    if accs.ndim == 1:
        init_dis, init_spd = initial_states[0], initial_states[1]
        accs = accs.reshape(-1)
        pred_spd = init_spd + np.cumsum(accs, axis=0) * dt
        pred_dis = init_dis + np.cumsum(pred_spd * dt, axis=0)
        return np.vstack([pred_dis, pred_spd, accs]).T
    if accs.ndim != 2:
        raise ValueError("accs must be 1D or 2D with shape (N, T)")

    initial_states = np.asarray(initial_states)
    if initial_states.ndim != 2 or initial_states.shape[1] != 2:
        raise ValueError("initial_states must have shape (N, 2)")

    init_dis = initial_states[:, 0:1]
    init_spd = initial_states[:, 1:2]
    pred_spd = init_spd + np.cumsum(accs, axis=1) * dt
    pred_dis = init_dis + np.cumsum(pred_spd * dt, axis=1)
    return np.stack([pred_dis, pred_spd, accs], axis=2)
