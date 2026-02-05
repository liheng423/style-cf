from os import name
from typing import List
from sympy import true
from tensordict import TensorDict
import torch.nn.functional as F
import torch
from src.schema import CFNAMES as CF


class LossUtils:
    @staticmethod
    def _predict_kinematics(accs: torch.Tensor, ground_truth: torch.Tensor, dt: float):
        """
        Compute predicted speed and distance based on acceleration.
        
        Args:
            accs (torch.Tensor): Acceleration tensor of shape (N, T).
            ground_truth (torch.Tensor): Ground truth tensor of shape (N, T+1, [distance, speed, acc]).
            dt (float): Time step duration.
        
        Returns:
            pred_spd (torch.Tensor): Predicted speed tensor of shape (N, T).
            pred_dis (torch.Tensor): Predicted distance tensor of shape (N, T).
        """
        init_dis = ground_truth[:, 0, 0].unsqueeze(1)
        init_spd = ground_truth[:, 0, 1].unsqueeze(1)
        pred_spd = init_spd + torch.cumsum(accs, dim=1) * dt
        # pred_dis = init_dis + torch.cumsum(pred_spd * dt + 0.5 * accs * dt**2, dim=1)
        pred_dis = init_dis + torch.cumsum(pred_spd, dim=1) * dt
        return pred_spd, pred_dis

class LossFunctions:

    @staticmethod
    def acc_spacing_mse(accs: torch.Tensor, ground_truth: torch.Tensor, dt: float, true_deltax_idx: int, true_leadx_idx: int):
        """
        Args:
            accs (torch.Tensor): Acceleration tensor of shape (N, T).
            ground_truth (torch.Tensor):  Ground truth tensor of shape (N, T+1, [distance, speed, acc, ...]).
            dt (float): Time step duration.
        """


        _, pred_dis = LossUtils._predict_kinematics(accs, ground_truth, dt)
        true_lead_x = ground_truth[:, :, true_leadx_idx]
        true_delta_x = ground_truth[:, :, true_deltax_idx]
        return F.mse_loss(true_lead_x - pred_dis, true_delta_x)
    
    @staticmethod
    def acc_dis_mse(accs: torch.Tensor, ground_truth: torch.Tensor, dt: float):


        _, pred_selfx = LossUtils._predict_kinematics(accs, ground_truth, dt)
        true_selfx = ground_truth[:, :, 0]  # (N, T)
        return F.mse_loss(pred_selfx, true_selfx)


###### Data Specific Loss Functions ######

class StyleLoss:

    def __init__(self, y_features: List[str]):
        
        self.name_dict = {feat: idx for idx, feat in enumerate(y_features)}
        assert CF.DELTA_X in self.name_dict
        assert CF.LEAD_X in self.name_dict

        

    def acc_spacing_mse(self, outputs: torch.Tensor, y: TensorDict, dt):
        if isinstance(outputs, tuple):
            output_accs = outputs[0]
        else:
            output_accs = outputs
        y_traj = y.get("y_seq").rename(None)
        acc_loss = LossFunctions.acc_spacing_mse(output_accs, y_traj, dt, self.name_dict[CF.DELTA_X], self.name_dict[CF.LEAD_X])
        return acc_loss

class IDMLoss:

    def __init__(self):
        pass

    def acc_dis_mse(self, outputs: torch.Tensor, y: torch.Tensor, dt):
        ### since IDM works individually, there's no batch dimension
        outputs = outputs.unsqueeze(0)
        accs_outputs = outputs[..., 2]
        y = y.unsqueeze(0)
        return LossFunctions.acc_dis_mse(accs_outputs, y, dt)
        


    
