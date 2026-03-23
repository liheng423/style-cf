from __future__ import annotations

import torch
import torch.nn.functional as F


def predict_kinematics(accs: torch.Tensor, ground_truth: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate acceleration into speed and distance with ground-truth initial state.
    """
    init_dis = ground_truth[:, 0, 0].unsqueeze(1)
    init_spd = ground_truth[:, 0, 1].unsqueeze(1)

    pred_spd = init_spd + torch.cumsum(accs, dim=1) * dt
    pred_dis = init_dis + torch.cumsum(pred_spd, dim=1) * dt
    return pred_spd, pred_dis


def spacing_mse_from_acc(
    accs: torch.Tensor,
    ground_truth: torch.Tensor,
    dt: float,
    true_deltax_idx: int,
    true_leadx_idx: int,
) -> torch.Tensor:
    _, pred_dis = predict_kinematics(accs, ground_truth, dt)
    t_len = accs.shape[1]
    true_lead_x = ground_truth[:, 1 : t_len + 1, true_leadx_idx]
    true_delta_x = ground_truth[:, 1 : t_len + 1, true_deltax_idx]
    return F.mse_loss(true_lead_x - pred_dis, true_delta_x)


def acc_mse_from_acc(
    accs: torch.Tensor,
    ground_truth: torch.Tensor,
    true_acc_idx: int,
) -> torch.Tensor:
    t_len = accs.shape[1]
    true_acc = ground_truth[:, 1 : t_len + 1, true_acc_idx]
    return F.mse_loss(accs, true_acc)


def distance_mse_from_acc(accs: torch.Tensor, ground_truth: torch.Tensor, dt: float) -> torch.Tensor:
    _, pred_selfx = predict_kinematics(accs, ground_truth, dt)
    t_len = accs.shape[1]
    true_selfx = ground_truth[:, 1 : t_len + 1, 0]
    return F.mse_loss(pred_selfx, true_selfx)
