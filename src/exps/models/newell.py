"""
Newell's car-following model (time-space shift model):
- Follower trajectory is a time-lagged (reaction time τ) and space-shifted
  (minimum spacing s) version of the leader trajectory.
- In discrete simulation, follower speed follows the leader's lagged speed,
  capped by a free-flow speed, while enforcing a minimum spacing.
"""

import torch


class NewellModel:
    """
    Single-vehicle Newell model with a single-step update.
    """

    def __init__(self, dt: float, params: dict[str, float]):
        self.dt = dt
        self.params = params

    def step(
        self,
        step_idx: int,
        ref_movements: torch.Tensor,
        prev_pos: torch.Tensor,
        prev_speed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step update for a single follower vehicle.

        Args:
            step_idx: Current time index (t).
            ref_movements: Leader movement, shape (T, 3), columns [x, v, a].
            prev_pos: Follower position at t-1.
            prev_speed: Follower speed at t-1.

        Returns:
            tuple: (pos_t, speed_t, acc_t) at time t.
        """
        ref_movements = torch.as_tensor(ref_movements)
        if isinstance(prev_pos, torch.Tensor):
            prev_pos = prev_pos.to(device=ref_movements.device, dtype=ref_movements.dtype)
        else:
            prev_pos = torch.tensor(prev_pos, device=ref_movements.device, dtype=ref_movements.dtype)
        if isinstance(prev_speed, torch.Tensor):
            prev_speed = prev_speed.to(device=ref_movements.device, dtype=ref_movements.dtype)
        else:
            prev_speed = torch.tensor(prev_speed, device=ref_movements.device, dtype=ref_movements.dtype)

        leader_x_now = ref_movements[step_idx, 0]
        lagged_time_idx = max(0, int(step_idx - self.params["reaction_time"] / self.dt))
        leader_speed_lagged = ref_movements[lagged_time_idx, 1]

        desired_speed = torch.clamp(
            leader_speed_lagged,
            min=0.0,
            max=self.params["freeflow_spd"],
        )

        potential_pos = prev_pos + desired_speed * self.dt
        safe_pos_limit = leader_x_now - self.params["veh_length"] - self.params["min_spacing"]
        pos_t = torch.minimum(potential_pos, safe_pos_limit)

        speed_t = torch.clamp((pos_t - prev_pos) / self.dt, min=0.0)
        acc_t = (speed_t - prev_speed) / self.dt

        return pos_t, speed_t, acc_t
