from ast import Not
from sklearn.discriminant_analysis import StandardScaler
import torch
from typing import Optional, List
from torch import Tensor
import numpy as np

@staticmethod
def _predict_kinematics(accs: torch.Tensor, initial_states: torch.Tensor, dt: float):
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
def _predict_kinematics_np(accs: np.ndarray, initial_states: np.ndarray, dt: float):
    """
    NumPy version of kinematics prediction.

    Args:
        accs (np.ndarray): Acceleration array of shape (T,).
        initial_states (np.ndarray): Initial [displacement, speed].
        dt (float): Time step duration.
    """
    init_dis, init_spd = initial_states[0], initial_states[1]
    accs = np.asarray(accs).reshape(-1)
    pred_spd = init_spd + np.cumsum(accs, axis=0) * dt
    pred_dis = init_dis + np.cumsum(pred_spd * dt, axis=0)
    return np.vstack([pred_dis, pred_spd, accs]).T


@staticmethod
def _predict_kinematics_np_batch(accs: np.ndarray, initial_states: np.ndarray, dt: float):
    """
    NumPy version of kinematics prediction for batched data.

    Args:
        accs (np.ndarray): Acceleration array of shape (N, T).
        initial_states (np.ndarray): Initial [displacement, speed] of shape (N, 2).
        dt (float): Time step duration.
    Return:
        np.ndarray: Predicted kinematics of shape (N, T, 3).
    """
    accs = np.asarray(accs)
    if accs.ndim == 1:
        return _predict_kinematics_np(accs, initial_states, dt)
    if accs.ndim != 2:
        raise ValueError("accs must be 1D or 2D with shape (N, T)")

    initial_states = np.asarray(initial_states)
    if initial_states.ndim != 2 or initial_states.shape[1] != 2:
        raise ValueError("initial_states must have shape (N, 2)")

    init_dis = initial_states[:, 0]
    init_spd = initial_states[:, 1]
    pred_spd = init_spd + np.cumsum(accs, axis=1) * dt
    pred_dis = init_dis + np.cumsum(pred_spd * dt, axis=1)
    return np.stack([pred_dis, pred_spd, accs], axis=2)


class Agent:

    def __init__(self, cf_model, dt: float, pred_horizon: int, historic_step: int, scaler: Optional[StandardScaler], 
                 pred_speed: bool = False, start_timestep=0):
        """
        This class only accepts the trained/calibrated car-following model.
        And used for closed-loop (recursive) prediction for a long period
        """
        self.dt = dt
        self.cf_model = cf_model
        self.pred_horizon: int = pred_horizon
        self.historic_step: int = historic_step
        self.scaler = scaler
        self.real_pred_step: int = pred_horizon
        self.start_time = start_timestep
        
        self.if_pred_speed: bool = pred_speed # True -> output is speed, False -> outout is Acceleration

    def _predict_onestep(self, data: torch.Tensor, initial_states, pred_func, if_last):
        pred = pred_func(self.cf_model, data, if_last)
        pred_series = _predict_kinematics(pred, initial_states, self.dt)
        return pred_series

    def _update_train_series(self, train_series: torch.Tensor, self_movements: torch.Tensor, leader_movements: torch.Tensor):
        """
        Implement how to update the training series with predicted self movements and leader movements.
        """
        raise NotImplementedError("This function must be rewritten to use")
    
    @staticmethod
    def _concat(tensor_list: List[Tensor]):
        """
        Implement how to stack items in the list along time dimension.
        """
        return NotImplementedError("This function must be rewritten to use")

    def predict(self, x_series: torch.Tensor, y_self_series: torch.Tensor, y_leader_series: torch.Tensor, pred_func = lambda model, data: model(data), mask = lambda x, n: x):
        """
        Args:
            x_series: np.array (time, input_features]) for training
            y_series: np.array (time, [x_self, v_self, a_self, x_leader, v_leader, a_leader]) for evaluation and visualization
            pred_func: function
        """
        
        assert y_self_series.shape[1] == 3 and y_leader_series.shape[1] == 3

        start_time = self.start_time

        skipped_movements = y_self_series[:start_time]

        y_self_series = y_self_series[start_time:]
        y_leader_series = y_leader_series[start_time:]
        x_series = x_series[start_time:]

        num_step = int((y_self_series.shape[0] - self.historic_step - self.pred_horizon + self.real_pred_step) // self.real_pred_step)


        x_series_train = x_series[:self.historic_step]
        self_movements = y_self_series[:self.historic_step] # only (time, [x_self, v_self, a_self])

        pred_time_start = self.historic_step

        for step in range(num_step):

            # train_time_window = slice(step * (self.pred_horizon), pred_time_start + step * (self.pred_horizon))
            # time window for prediction horizon
            pred_time_window = slice(pred_time_start + step * (self.real_pred_step), pred_time_start + (step) * (self.real_pred_step) + self.pred_horizon)

            # time window for update used prediction
            real_pred_window = slice(pred_time_start + step * (self.real_pred_step), pred_time_start + (step + 1) * (self.real_pred_step))

            with torch.no_grad():
                # data = x_series_train[-self.pred_horizon:]

                # We use train_time_window (seq_len) and pred window (pred_len) as data here, but user-defined mask function
                # could ignore the second param
                data = mask(x_series_train[-self.historic_step:], x_series[pred_time_window], x_series[:self.historic_step])


                # predict the acceleration, speed and distance
                pred_series = self._predict_onestep(data, self_movements[-1, [0, 1]], pred_func, step == num_step-1)

                # update self_movements and x_series_train
                if step == num_step-1:
                    x_series_train = self._concat([x_series_train, self._update_train_series(x_series[pred_time_window], pred_series, y_leader_series[pred_time_window])])
                else:
                    x_series_train = self._concat([x_series_train, self._update_train_series(x_series[real_pred_window], pred_series, y_leader_series[real_pred_window])])
                self_movements = self._concat([self_movements, pred_series])

        return self._concat([skipped_movements, self_movements])
