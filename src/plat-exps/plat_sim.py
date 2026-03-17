from __future__ import annotations

from src.exps.models.simulation import vstack
import torch
from src.exps.agent import Agent
from src.exps.models.newell import NewellModel
from typing import Any, Callable, Optional, Protocol, Sequence, cast


class TrainSeries(Protocol):
    """
    Minimal protocol for the training series objects used by the simulator.
    """

    inputs: tuple[Any, Any, Any]

    def extend(self, total_timesteps: int) -> "TrainSeries":
        ...

    def replace(self, sl: slice, value: "TrainSeries") -> "TrainSeries":
        ...

    def __getitem__(self, sl: slice) -> "TrainSeries":
        ...


class Env:
    
    def __init__(
        self,
        platoon: Sequence[Agent],
        header_movements: torch.Tensor,
        dt: float,
        pred_funcs: Sequence[Any],
        masks: Sequence[Any],
        pred_duration: float,
        hist_duration: float,
        dummy_train_sers: Sequence[TrainSeries],
        veh_lens: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            platoon: Ordered list of vehicles (leader first).
            header_movements: Leader trajectory, shape (T, 3), columns [x, v, a].
            dt: Simulation time step (s).
            pred_funcs: Per-vehicle prediction functions.
            masks: Per-vehicle prediction masks.
            pred_duration: Prediction horizon duration (s).
            hist_duration: History window duration (s).
            dummy_train_sers: Per-vehicle training series templates.
            veh_lens: Per-vehicle length/spacing thresholds. Defaults to 8 m each.
        """
        self.header_movements = header_movements
        self.header = platoon[0]
        self.platoon = platoon
        self.platoon_len = len(platoon)
        self.dt = dt
        self.pred_funcs = pred_funcs
        self.masks = masks
        self.veh_lens = [8.0 for _ in range(len(platoon))] if veh_lens is None else list(veh_lens)

        # All models must use the same prediction step.
        self.pred_duration = pred_duration
        self.hist_duration = hist_duration

        # assert dummy_train_series[1].shape[0] == 1
        # assert len(dummy_train_series[1].shape) == 2
        # (1, features) but only non-trajectory features are required.
        self.dummy_train_sers = dummy_train_sers

    def _generate_start(
        self,
        gen_step: int,
        ref_movements: torch.Tensor,
        newell_dict: dict[str, float],
    ) -> torch.Tensor:
        """
        Generate a follower trajectory using Newell's car-following model.

        Args:
            gen_step: Number of time steps to generate.
            ref_movements: Leader movements, shape (T, 3), columns [x, v, a].
            newell_dict: Parameters for Newell's model.

        Returns:
            torch.Tensor: Follower movements, shape (T, 3), columns [x, v, a].
        """

        model = NewellModel(self.dt, newell_dict)
        num_steps = gen_step
        follower_positions = torch.zeros(num_steps)
        follower_speeds = torch.zeros(num_steps)
        follower_accs = torch.zeros(num_steps)

        init_spacing = newell_dict["init_spacing"]
        follower_positions[0] = ref_movements[0, 0] - init_spacing
        follower_speeds[0] = ref_movements[0, 1]
        follower_accs[0] = 0.0

        for i in range(1, num_steps):
            pos_t, speed_t, acc_t = model.step(
                step_idx=i,
                ref_movements=ref_movements,
                prev_pos=follower_positions[i - 1],
                prev_speed=follower_speeds[i - 1],
            )
            follower_positions[i] = pos_t
            follower_speeds[i] = speed_t
            follower_accs[i] = acc_t

        follower_movements = torch.stack([follower_positions, follower_speeds, follower_accs], axis=1)
        return follower_movements
    
    def _force_positive_v(self, movement: torch.Tensor, min_speed: float = 0.01) -> torch.Tensor:
        """
        Replace any negative speeds with min_speed and recompute a and x.

        Args:
            movement: Shape (T, 3), columns [position, speed, acceleration].
            min_speed: Minimum allowed speed (m/s).

        Returns:
            torch.Tensor: Corrected movement.
        """
        movement = movement.clone()  # Avoid in-place mutation of the input.

        if torch.all(movement[:, 1] >= min_speed):
            return movement

        # Clamp speeds to a physically valid minimum
        v = torch.clamp(movement[:, 1], min=min_speed)
        x0 = movement[0, 0]

        # Recompute acceleration from velocity: a_t = (v_t - v_{t-1}) / dt
        a = torch.gradient(v, spacing=self.dt)[0]
        movement[:, 2] = a

        # Recompute position from velocity: x_t = x0 + cumsum(v * dt)
        x = x0 + torch.cumsum(v * self.dt, dim=0)
        movement[:, 0] = x

        movement[:, 1] = v
        assert torch.all(movement[:, 1] >= min_speed)

        return movement
            


    def _initialize_simulation(
        self,
        total_timesteps: int,
        newell_dict: dict[str, float],
    ) -> tuple[list[torch.Tensor], list[TrainSeries | None], list[TrainSeries | None], torch.Tensor]:
        movements: list[torch.Tensor] = [
            self.header_movements[:int(self.hist_duration / self.dt)].clone()
        ]
        x_sers: list[TrainSeries | None] = [None]
        x_train_sers: list[TrainSeries | None] = [None for _ in range(self.platoon_len)]
        groundtruth_movements = torch.zeros([self.platoon_len, total_timesteps, 3])
        groundtruth_movements[0, :] = self.header_movements[:total_timesteps]

        for veh_idx in range(1, self.platoon_len):
            dummy_x_ser = self.dummy_train_sers[veh_idx].extend(total_timesteps)
            groundtruth_movements[veh_idx, :, :] = self._generate_start(total_timesteps, groundtruth_movements[veh_idx - 1, :, :], newell_dict)
            update_train_series = cast(
                Callable[[TrainSeries, torch.Tensor, torch.Tensor], TrainSeries],
                self.platoon[veh_idx]._update_train_series,
            )
            x_ser = update_train_series(
                dummy_x_ser,
                groundtruth_movements[veh_idx, :],
                groundtruth_movements[veh_idx - 1, :],
            )
            x_sers.append(x_ser)
            x_train_sers[veh_idx] = x_ser[:int(self.hist_duration / self.dt)]
            movements.append(groundtruth_movements[veh_idx, :int(self.hist_duration / self.dt)].clone())

        return movements, x_sers, x_train_sers, groundtruth_movements
    
    def _force_no_collision(
        self,
        movement: torch.Tensor,
        leader_movement: torch.Tensor,
        min_spacing: float,
    ) -> torch.Tensor:
        """
        If spacing to the leader is below min_spacing, force v=0 and recompute a and x.

        Args:
            movement: Shape (T, 3), columns [position, speed, acceleration].
            leader_movement: Shape (T, 3), columns [position, speed, acceleration].
            min_spacing: Minimum allowed spacing.

        Returns:
            torch.Tensor: Corrected movement.
        """
        movement = movement.clone()
        v = movement[:, 1].clone()
        x0 = movement[0, 0]
        leader_x = leader_movement[:, 0]

        # Check spacing per frame and force v=0 if too close.
        for t in range(len(v)):
            spacing = leader_x[t] - x0 - torch.sum(v[:t] * self.dt) if t > 0 else leader_x[t] - x0
            if spacing < min_spacing:
                v[t] = 0.0  # Force stop.

        # Recompute acceleration from corrected velocity.
        a = torch.gradient(v, spacing=self.dt)[0]

        # Recompute position: x_t = x0 + sum(v * dt).
        x = x0 + torch.cumsum(v * self.dt, dim=0)

        # Update movement.
        movement[:, 0] = x
        movement[:, 1] = v
        movement[:, 2] = a

        return movement

    def _predict_with_agent(
        self,
        agent: Agent,
        x_ser: TrainSeries,
        x_train_ser: TrainSeries,
        self_movements: torch.Tensor,
        leader_movements: torch.Tensor,
        pred_time_window: slice,
        pred_func: Any,
        mask: Any,
    ) -> torch.Tensor:
        hist_step = agent.historic_step
        horizon_len = agent.horizon_len

        self_hist = self_movements[-hist_step:]
        leader_hist = leader_movements[-hist_step:]
        leader_future = leader_movements[pred_time_window]

        x_hist = x_train_ser[-hist_step:]
        x_future = x_ser[pred_time_window]
        x_full = vstack([x_hist, x_future])

        self_traj_full = torch.cat(
            [self_hist, torch.zeros_like(leader_future)],
            dim=0,
        )
        leader_traj_full = torch.cat([leader_hist, leader_future], dim=0)

        pred_full = agent.predict(x_full, self_traj_full, leader_traj_full, pred_func, mask)
        return pred_full[-horizon_len:]

    def _predict_step(
        self,
        step: int,
        pred_time_window: slice,
        x_sers: list[TrainSeries | None],
        x_train_sers: list[TrainSeries | None],
        movements: list[torch.Tensor],
        active_mask: Optional[list[bool]] = None,
    ) -> None:
        for veh_idx in range(1, self.platoon_len):
            if active_mask is not None and not active_mask[veh_idx]:
                continue

            mask = self.masks[veh_idx]
            agent = self.platoon[veh_idx]
            pred_func = self.pred_funcs[veh_idx]
            x_train_ser = x_train_sers[veh_idx]
            x_ser = x_sers[veh_idx]
            if x_train_ser is None or x_ser is None:
                raise ValueError(f"Missing training series for vehicle {veh_idx}.")

            with torch.no_grad():
                pred_series = self._predict_with_agent(
                    agent=agent,
                    x_ser=x_ser,
                    x_train_ser=x_train_ser,
                    self_movements=movements[veh_idx],
                    leader_movements=movements[veh_idx - 1],
                    pred_time_window=pred_time_window,
                    pred_func=pred_func,
                    mask=mask,
                )
                pred_series = self._force_positive_v(pred_series, min_speed=0)
                if veh_idx > 0:
                    pred_series = self._force_no_collision(pred_series, movements[veh_idx - 1][pred_time_window], self.veh_lens[veh_idx])
                update_train_series = cast(
                    Callable[[TrainSeries, torch.Tensor, torch.Tensor], TrainSeries],
                    agent._update_train_series,
                )
                x_train_ser_pred = update_train_series(
                    x_ser[pred_time_window],
                    pred_series,
                    movements[veh_idx - 1][pred_time_window],
                )
                x_train_ser = vstack([x_train_ser, x_train_ser_pred])
                x_train_sers[veh_idx] = x_train_ser

                if veh_idx + 1 < self.platoon_len and (active_mask is None or active_mask[veh_idx + 1]):
                    x_fo_ser = x_sers[veh_idx + 1]
                    if x_fo_ser is None:
                        raise ValueError(f"Missing training series for vehicle {veh_idx + 1}.")
                    update_train_series_lead = cast(
                        Callable[[TrainSeries, torch.Tensor], TrainSeries],
                        self.platoon[veh_idx + 1]._update_train_series_lead,
                    )
                    x_fo_ser_pred_window = update_train_series_lead(
                        x_fo_ser[pred_time_window],
                        pred_series,
                    )
                    x_sers[veh_idx + 1] = x_fo_ser.replace(pred_time_window, x_fo_ser_pred_window)

                movements[veh_idx] = vstack([movements[veh_idx], pred_series])

                if active_mask is not None:
                    if pred_series[-1, 0] < self.x_range[0] or pred_series[-1, 0] > self.x_range[1]:
                        active_mask[veh_idx] = False

    def run_until_exit(self, x_range: tuple[float, float], newell_dict: dict[str, float]) -> list[torch.Tensor]:
        """
        Simulate until all followers exit x_range.

        - Extend leader data on demand with constant-velocity extrapolation.
        - Before each predict step, ensure x_sers[1][pred_window] exists by
          calling agent._update_train_series_lead with leader movements.
        - Downstream vehicles' x_sers are updated inside _predict_step.
        - On return, NaN out any extrapolated leader tail that exceeds the finish line.
        """
        self.x_range = x_range
        finish_line = x_range[1]

        device = self.header_movements.device
        dtype  = self.header_movements.dtype

        pred_step = int(self.pred_duration / self.dt)
        hist_step = int(self.hist_duration / self.dt)

        # Record original leader length to mark extrapolated tail on return.
        orig_len = self.header_movements.shape[0]

        # Track follower activeness only; leader is always active to avoid deadlock.
        active_mask = [False] + [True] * (self.platoon_len - 1)

        # Initialize ground truth and features using current leader length.
        init_len = orig_len
        movements, x_sers, x_train_sers, _ = self._initialize_simulation(init_len, newell_dict)

        t = hist_step

        # -------- helper: ensure leader long enough (constant-velocity extrapolation) --------
        def _ensure_header_len(need_len: int):
            cur_len = self.header_movements.shape[0]
            if need_len <= cur_len:
                return
            steps = need_len - cur_len
            dt = self.dt

            x_last = self.header_movements[-1, 0]
            v_last = self.header_movements[-1, 1]

            ks   = torch.arange(1, steps + 1, device=device, dtype=dtype)
            xext = x_last + v_last * dt * ks
            vext = torch.full((steps,), v_last, device=device, dtype=dtype)
            aext = torch.zeros(steps, device=device, dtype=dtype)

            ext = torch.stack([xext, vext, aext], dim=1)  # (steps, 3)
            self.header_movements = torch.cat([self.header_movements, ext], dim=0)

        # -------- main loop: run until all followers exit --------
        while any(active_mask[1:]):
            pred_time_window = slice(t, t + pred_step)

            # 1) Ensure leader data covers this window; append to movements[0].
            _ensure_header_len(t + pred_step)
            movements[0] = torch.cat([movements[0], self.header_movements[pred_time_window]], dim=0)

            # 2) Precompute x_ser for follower-1 (others cascade inside _predict_step).
            if self.platoon_len > 1 and (active_mask[1]):
                x_fo_ser = x_sers[1]
                if x_fo_ser is None:
                    raise ValueError("Missing training series for vehicle 1.")
                # Take the window view and inject leader speed into decoder inputs.
                x_fo_ser_win = x_fo_ser[pred_time_window]
                leader_win = movements[0][pred_time_window]  # Leader window
                x_fo_ser_win = self.platoon[1]._update_train_series_lead(
                    x_fo_ser_win,
                    leader_win,
                )
                # Write back (requires MultiDataSingle.replace to preserve behavior).
                x_sers[1] = x_fo_ser.replace(pred_time_window, x_fo_ser_win)

            # 3) Predict the window:
            #    - Build inputs using mask(...)
            #    - Predict per-vehicle series and append to movements
            #    - Cascade update x_sers[veh_idx+1][pred_window]
            #    - Maintain active_mask (out of range -> False)
            self._predict_step(-1, pred_time_window, x_sers, x_train_sers, movements, active_mask)

            t += pred_step

        # -------- cleanup: NaN out extrapolated leader tail past finish line --------
        if movements[0].numel() > 0:
            x0 = movements[0][:, 0]
            imagined_mask = torch.zeros_like(x0, dtype=torch.bool, device=movements[0].device)
            imagined_mask[orig_len:] = True  # Mark extrapolated tail only.
            beyond_finish = x0 >= finish_line
            mask = imagined_mask & beyond_finish
            if mask.any():
                movements[0][mask] = torch.full(
                    (3,), torch.nan, device=movements[0].device, dtype=movements[0].dtype
                )

        return movements


    def start(self, sim_time: float, newell_dict: dict[str, float]) -> list[torch.Tensor]:
        total_timesteps = int(sim_time / self.dt)
        num_step = int((sim_time - self.hist_duration) / self.pred_duration)
        pred_step = int(self.pred_duration / self.dt)
        hist_step = int(self.hist_duration / self.dt)

        movements, x_sers, x_train_sers, _ = self._initialize_simulation(total_timesteps, newell_dict)
        pred_time_start = hist_step

        for step in range(num_step):
            pred_time_window = slice(pred_time_start + step * pred_step, pred_time_start + (step + 1) * pred_step)
            movements[0] = vstack([movements[0], self.header_movements[pred_time_window]])
            self._predict_step(step, pred_time_window, x_sers, x_train_sers, movements)

        return movements
