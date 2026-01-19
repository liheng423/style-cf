from tensordict import TensorDict
import torch.nn.functional as F
import torch
from src.models.agent import _predict_kinematics

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
        init_dis, init_spd = ground_truth[:, 0, 0].unsqueeze(1), ground_truth[:, 0, 1].unsqueeze(1)
        pred_spd = init_spd + torch.cumsum(accs, dim=1) * dt
        # pred_dis = init_dis + torch.cumsum(pred_spd * dt + 0.5 * accs * dt**2, dim=1)
        pred_dis = init_dis + torch.cumsum(pred_spd , dim=1) * dt
        return pred_spd, pred_dis

class LossFunctions:

    @staticmethod
    def acc_spacing_mse(accs: torch.Tensor, ground_truth: torch.Tensor, dt: float):
        _, pred_dis = LossUtils._predict_kinematics(accs, ground_truth, dt)
        true_leader_dis = ground_truth[:, 1:, 4]
        true_spacing = ground_truth[:, 1:, 3]
        return F.mse_loss(true_leader_dis - pred_dis, true_spacing)

###### LOSS FUNCTION ######
class StyleLoss:

    @staticmethod
    def acc_spacing_mse(outputs: torch.Tensor, y: TensorDict, dt):
                    
        output_accs, output_style = outputs
        y_traj = y.get("y_seq").rename(None)
        acc_loss = LossFunctions.acc_spacing_mse(output_accs, y_traj, dt)
        return acc_loss