from ast import Not
from typing import Callable, Optional, List, Any

import numpy as np
import torch
from sklearn.discriminant_analysis import StandardScaler
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module

from src.models.kine_utils import _predict_kinematics
from src.models.utils import SliceableTensorDict
from src.stylecf.schema import TensorNames



class Agent:

    def __init__(
        self,
        cf_model: Module,
        dt: float,
        horizon_len: int,
        historic_step: int,
        scaler: Optional[StandardScaler],
        pred_speed: bool = False,
        start_timestep: int = 0,
    ) -> None:
        """
        This class only accepts the trained/calibrated car-following model.
        And used for closed-loop (recursive) prediction for a long period

        Args:
            cf_model: Trained/calibrated car-following model used for prediction.
            dt: Simulation time step in seconds.
            horizon_len: Number of steps predicted in each horizon window.
            historic_step: Number of past steps used as model context.
            scaler: Optional scalers used for input/output normalization.
            start_timestep: Index to start prediction from in the input series.
        """
        self.dt = dt
        self.cf_model = cf_model
        self.horizon_len: int = horizon_len # prediction window length of the model
        self.historic_step: int = historic_step # 
        self.scaler = scaler
        self.rollout_step: int = horizon_len # actual rollout step 
        self.start_step: int = start_timestep


    def _predict_onestep(
        self,
        data: Tensor,
        initial_states: Tensor,
        pred_func: Callable[[Module, Any, bool], Tensor],
        if_last: bool,
    ) -> Tensor:
        pred = pred_func(self.cf_model, data, if_last).rename(None)
        pred_traj = _predict_kinematics(pred, initial_states.rename(None), self.dt)
        return pred_traj

    def _update_train_series(
        self,
        train_series: Tensor,
        self_movements: Tensor,
        leader_movements: Tensor,
    ) -> Tensor:
        """
        Implement how to update the training series with predicted self movements and leader movements.
        """
        raise NotImplementedError("This function must be rewritten to use")

    
    @staticmethod
    def _concat(tensor_list: List[Tensor | TensorDict]):
        """
        Implement how to stack items in the list along time dimension.
        tensor_list: List[torch.Tensor], each tensor in the shape of (time, [distance, velocity, acceleration])
        """
        return NotImplementedError("This function must be rewritten to use")
    

    def predict(
        self,
        x_full: SliceableTensorDict,
        self_traj_full: Tensor,
        leader_traj_full: Tensor,
        pred_func: Callable[[Module, Any, bool], Tensor] = lambda model, data, *_: model(data),
        mask: Callable[[SliceableTensorDict, SliceableTensorDict, SliceableTensorDict], Any] = lambda x, *_: x,
    ) -> Tensor:
        """
        Args:
            x_full: TensorDict, full model input series (for model prediction)
            self_traj_full: torch.Tensor (time, [x_self, v_self, a_self]) full self trajectory 
            leader_traj_full: torch.Tensor (time, [x_leader, v_leader, a_leader]) full leader trajectory
            pred_func: function
            mask: function to mask the input data for model prediction
        """
        
        assert self_traj_full.shape[1] == 3 and leader_traj_full.shape[1] == 3

        start_step = self.start_step

        skipped_movements = self_traj_full[:start_step]

        self_traj_full = self_traj_full[start_step:]
        leader_traj_full = leader_traj_full[start_step:]
        x_full = x_full.sel(T=slice(start_step, None))

        num_step = int((self_traj_full.shape[0] - self.historic_step - self.horizon_len + self.rollout_step) // self.rollout_step)


        x_ctx = x_full.sel(T=slice(None, self.historic_step))
        self_movements = self_traj_full[:self.historic_step] # only (time, [x_self, v_self, a_self])

        horizon_start = self.historic_step

        for step in range(num_step):
            base = horizon_start + step * self.rollout_step

            # time window for full prediction horizon
            horizon_window = slice(base, base + self.horizon_len)

            # time window for the actual rollout step
            rollout_window = slice(base, base + self.rollout_step)

            with torch.no_grad():


                # We use train_time_window (seq_len) and pred window (pred_len) as data here, but user-defined mask function
                # could ignore the second param
                data = mask(x_ctx.sel(T=slice(-self.historic_step, None)), 
                            x_full.sel(T=horizon_window), 
                            x_full.sel(T=slice(None, self.historic_step)))


                # predict the acceleration, speed and distance
                pred_traj = self._predict_onestep(data, self_movements[-1, [0, 1]], pred_func, step == num_step-1)

                # update self_movements and x_ctx
                update_window = horizon_window if step == num_step - 1 else rollout_window
                x_ctx = self._concat([x_ctx, self._update_train_series(x_full.sel(T=update_window), pred_traj, leader_traj_full[update_window])])
                self_movements = self._concat([self_movements, pred_traj])

        return self._concat([skipped_movements, self_movements])
