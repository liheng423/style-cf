import numpy as np
import pandas as pd
from typing import Optional, Sequence


class Wave:

    @staticmethod
    def wave_speed(
        movements: np.ndarray,
        decel_events_list: Sequence[np.ndarray],
        dt: float = 0.1,
        factor: float = 15.0,
    ) -> list[tuple[int, int, float, float, float, float, float]]:
        """
        Calculate deceleration wave speeds between consecutive vehicles.

        Args:
            movements (ndarray): Shape (N, T, ≥2), vehicle positions over time.
                                The first column in the last dimension is x-position.
            decel_events_list (list of ndarray): Each element is (num_events, 2)
                                                [start_time, end_time] for each vehicle.
            dt (float): Time step size in seconds. Default is 0.1.
            factor (float): Lookahead factor used as lookahead = factor / max(1, leader_speed).

        Returns:
            list of tuples: Each tuple is
                            (leader_idx, follower_idx, t_lead, t_foll, dx, dt_wave, v_wave)
                            where v_wave is estimated wave speed (m/s).
        """
        N, T, _ = movements.shape
        wave_speeds: list[tuple[int, int, float, float, float, float, float]] = []

        for i in range(N - 1):
            leader_events = decel_events_list[i]
            follower_events = decel_events_list[i + 1]

            for j in range(leader_events.shape[0]):
                t_lead = float(leader_events[j, 0])
                frame_lead = int(t_lead / dt)

                if frame_lead >= T:
                    continue

                # estimate current speed of leader
                current_speed = float(movements[i, frame_lead, 1])
                lookahead = factor / max(1.0, current_speed)

                x_lead = float(movements[i, frame_lead, 0])

                for k in range(follower_events.shape[0]):
                    t_foll = float(follower_events[k, 0])
                    if t_lead < t_foll <= t_lead + lookahead:
                        frame_foll = int(t_foll / dt)
                        if frame_foll >= T:
                            continue

                        x_foll = float(movements[i + 1, frame_foll, 0])

                        dx = x_lead - x_foll
                        dt_wave = t_lead - t_foll
                        v_wave = dx / dt_wave if dx > 0 else float("inf")

                        wave_speeds.append((i, i + 1, t_lead, t_foll, dx, dt_wave, v_wave))
                        break

        return wave_speeds
    

    @staticmethod
    def track_waves(
        wave_speeds: Sequence[tuple[int, int, float, float, float, float, float]],
        num_veh: int,
        time_tolerance: float = 1.0,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Track complete wave propagation paths from wave_speeds.

        Args:
            wave_speeds (list of tuples): From wave_speed(), each tuple is
                (from, to, t_lead, t_foll, dx, dt, wave_speed).
            num_veh (int): Total number of vehicles in the dataset.
            time_tolerance (float): Max allowed time difference (s) when matching
                                    waves between consecutive vehicle pairs.
            columns (list of str): Column names in order, e.g.
                ["wave_id", "from", "to", "t_lead", "t_foll", "dx", "dt", "wave_speed"]

        Returns:
            np.ndarray: Structured array with given columns.
        """
        if columns is None:
            raise ValueError("Must provide a list of column names.")

        used: set[tuple[int, int, float, float]] = set()
        wave_id = 0
        flat_records: list[list[float]] = []

        for w in wave_speeds:
            key = (int(w[0]), int(w[1]), float(w[2]), float(w[3]))
            if key in used:
                continue

            current_vehicle = int(w[1])
            current_time = float(w[3])
            flat_records.append([float(wave_id), float(w[0]), float(w[1]), float(w[2]), float(w[3]), float(w[4]), float(w[5]), float(w[6])])
            used.add(key)

            # Follow downstream vehicles
            while current_vehicle < num_veh - 1:
                candidates: list[tuple[int, int, float, float, float, float, float]] = [
                    w2 for w2 in wave_speeds
                    if w2[0] == current_vehicle
                    and abs(w2[2] - current_time) <= time_tolerance
                    and (w2[0], w2[1], w2[2], w2[3]) not in used
                ]
                if not candidates:
                    break

                next_w = min(candidates, key=lambda x: x[3])
                key = (int(next_w[0]), int(next_w[1]), float(next_w[2]), float(next_w[3]))
                used.add(key)
                flat_records.append([
                    float(wave_id),
                    float(next_w[0]),
                    float(next_w[1]),
                    float(next_w[2]),
                    float(next_w[3]),
                    float(next_w[4]),
                    float(next_w[5]),
                    float(next_w[6]),
                ])

                current_vehicle = int(next_w[1])
                current_time = float(next_w[3])

            wave_id += 1

        return np.array([tuple(row) for row in flat_records], dtype=float)

    def wave_velocity(self, wave_chain: pd.DataFrame, movements: np.ndarray, dt: float, time_ahead: float):

        avg_wave_speed = wave_chain["wave_speed"].mean(skipna=True)

        veh_speeds = [movements[int(row["to"]), int((row["t_lead"]  - time_ahead) / dt), 1] for _, row in wave_chain.iterrows()]
        
        avg_veh_speed = np.mean(veh_speeds)

        return avg_wave_speed, avg_veh_speed


class AmpFactor:
    @staticmethod
    def _local_median(arr_1d: np.ndarray, center_idx: int, halfw: int) -> float:
        n = arr_1d.shape[0]
        if n == 0:
            return np.nan
        left = max(0, center_idx - halfw)
        right = min(n - 1, center_idx + halfw)
        if left > right:
            return np.nan
        seg = arr_1d[left : right + 1]
        return float(np.median(seg)) if seg.size > 0 else np.nan

    @staticmethod
    def calc_amp_factor(
        movements: np.ndarray,
        wave_chain: np.ndarray,
        time_shift: float,
        dt: float = 0.1,
        flank_window_s: float = 0.5,
        tiny: float = 1e-2,
    ) -> np.ndarray:
        """
        Return one row:
        [wave_id, veh_leader, t_leader, veh_tail, t_tail, dv_leader, dv_tail, amp_factor]
        """
        if movements.ndim != 3 or movements.shape[2] < 2:
            raise ValueError("movements must have shape (num_veh, num_time, >=2)")
        if wave_chain.ndim != 2 or wave_chain.shape[1] < 8:
            raise ValueError("wave_chain must follow [wave_id, from, to, t_lead, t_foll, dx, dt, wave_speed]")

        speed = movements[:, :, 1]
        _, total_t = speed.shape
        shift_idx = int(round(time_shift / dt))
        halfw = max(0, int(round((flank_window_s / dt) / 2)))

        def clip_idx(idx: int) -> int:
            return int(np.clip(idx, 0, total_t - 1))

        rows = wave_chain[np.argsort(wave_chain[:, 3])]
        wave_id = int(rows[0, 0])

        veh_leader = int(rows[0, 1])
        t_leader_idx = int(round(rows[0, 3] / dt))
        veh_tail = int(rows[-1, 2])
        t_tail_idx = int(round(rows[-1, 4] / dt))

        if veh_leader < 0 or veh_tail < 0 or veh_leader >= speed.shape[0] or veh_tail >= speed.shape[0]:
            return np.empty((0, 8), dtype=float)

        t_lb = clip_idx(t_leader_idx - shift_idx)
        t_la = clip_idx(t_leader_idx + shift_idx)
        t_tb = clip_idx(t_tail_idx - shift_idx)
        t_ta = clip_idx(t_tail_idx + shift_idx)

        v_lb = AmpFactor._local_median(speed[veh_leader], t_lb, halfw)
        v_la = AmpFactor._local_median(speed[veh_leader], t_la, halfw)
        v_tb = AmpFactor._local_median(speed[veh_tail], t_tb, halfw)
        v_ta = AmpFactor._local_median(speed[veh_tail], t_ta, halfw)

        dv_leader = float(v_lb - v_la) if np.isfinite(v_lb) and np.isfinite(v_la) else np.nan
        dv_tail = float(v_tb - v_ta) if np.isfinite(v_tb) and np.isfinite(v_ta) else np.nan

        amp = np.nan
        if np.isfinite(dv_leader) and np.isfinite(dv_tail) and abs(dv_leader) > tiny:
            if np.sign(dv_leader) == np.sign(dv_tail):
                amp = abs(dv_tail) / abs(dv_leader)

        return np.array(
            [
                [
                    float(wave_id),
                    float(veh_leader),
                    float(rows[0, 3]),
                    float(veh_tail),
                    float(rows[-1, 4]),
                    dv_leader,
                    dv_tail,
                    amp,
                ]
            ],
            dtype=float,
        )
