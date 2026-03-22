import numpy as np
import warnings
from scipy.ndimage import gaussian_filter1d


class PlatoonMetrics:

    @staticmethod
    def avg_speed(plat_movements: np.ndarray):
        """
        Compute the average speed over all vehicles and all time steps.

        Args:
            plat_movements (np.ndarray): Array of shape [Veh, Time, 3],
                                         where the last dimension represents [x, v, a].

        Returns:
            float: The average speed.
        """
        # Extract the speed component (index 1 in the innermost dimension)
        speeds = plat_movements[:, :, 1]  # Shape: [Veh, Time]
        
        # Compute fleet-average speed (ignore NaNs if present)
        avg = np.nanmean(speeds)

        return avg


    @staticmethod
    def total_time_spent(plat_movements: np.ndarray, target_x: float, time_step: float = 0.1):
        """
        Compute Total Time Spent (TTS) inside the boundary x <= target_x across all vehicles.

        Args:
            plat_movements (np.ndarray): Shape [Veh, Time, 3], with [x, v, a].
            target_x (float): Downstream boundary; positions <= target_x are counted.
            time_step (float): Duration of each time step (s).

        Returns:
            float: Total Time Spent (veh*s) within the boundary.
        """
        x_positions = plat_movements[:, :, 0]  # [Veh, Time]
        speeds = plat_movements[:, :, 1]       # [Veh, Time]

        # Count time steps before crossing downstream boundary
        tts = np.sum(x_positions <= target_x) * time_step

        return float(tts)
    
    @staticmethod
    def total_stopped_time_last_vehicle(plat_movements: np.ndarray, speed_thres: float = 1.0, time_step: float = 0.1,
                                        L: float | None = None, offset: float = 700.0) -> float:
        """
        Compute total time (s) the last vehicle is below speed_thres before crossing the boundary.

        Args:
            plat_movements (np.ndarray): Shape [Veh, Time, 3], with [x, v, a].
            speed_thres (float): Threshold under which the vehicle is considered stopped.
            time_step (float): Duration of each time step (s).
            L (float | None): Downstream boundary. If None, use leader max position minus offset.
            offset (float): Offset subtracted from the leader max position when L is None.

        Returns:
            float: Total stopped time for the last vehicle (s).
        """
        x = plat_movements[..., 0]   # (N, T)
        N, T = x.shape
        leader = x[0]

        # Downstream boundary relative to leader max position
        if L is None:
            L = np.nanmax(leader) - offset

        

        # Only consider the last vehicle while it has not crossed the boundary
        last_v = plat_movements[-1, :, 1]  # [Time]
        stopped_mask = (last_v < speed_thres)
        no_crossing_mask = x[-1, :] < L
        stopped_mask = np.nan_to_num(stopped_mask, nan=False)

        total_stopped_steps = np.sum(stopped_mask & no_crossing_mask)
        return float(total_stopped_steps * time_step)
    

    @staticmethod
    def low_speed(plat_movements: np.ndarray, speed_thres: float = 1.0, time_step: float = 0.1, 
          L: float | None = None, offset: float = 700.0) -> float:
        """
        Calculate total low-speed time across all vehicles, used as a congestion indicator.
        
        Args:
            plat_movements (np.ndarray): shape [Veh, Time, 3], with [x, v, a].
            speed_thres (float): threshold under which vehicle is considered stopped (m/s).
            time_step (float): duration of each time step (in seconds).
            L (float | None): Downstream boundary. If None, use leader max position minus offset.
            offset (float): Offset subtracted from the leader max position when L is None.

        Returns:
            float: total low-speed time in veh*seconds.
        """
        x = plat_movements[..., 0]   # (N, T)
        N, T = x.shape
        leader = x[0]
        # Speed component (index 1 of [x, v, a])
        speeds = plat_movements[:, :, 1]

        # Low-speed mask across all vehicles and time
        low_speed_mask = speeds < speed_thres

        # Downstream boundary relative to leader max position
        if L is None:
            L = np.nanmax(leader) - offset

        no_crossing_mask = x < L


        # Count low-speed time steps per vehicle
        low_speed_counts = np.sum(low_speed_mask & no_crossing_mask, axis=1)

        # Total low-speed time (veh*s)
        total_low_speed_time = np.sum(low_speed_counts) * time_step

        return total_low_speed_time
    
    @staticmethod
    def delay(movements: np.ndarray, dt: float = 0.1, tau: float = 1.0, s: float = 7.5, 
          L: float | None = None, offset: float = 700.0) -> float:
        """
        Compute delay = TTS_actual - TTS_benchmark.
        
        Args:
            movements: (N, T, 3) array with columns [x, v, a].
            dt: time step.
            tau: time shift per vehicle in Newell benchmark.
            s: space shift per vehicle in Newell benchmark.
            L: downstream boundary; if None, use leader max position minus offset.
            offset: offset subtracted from leader max position.
        """
        x = movements[..., 0]   # (N, T)
        N, T = x.shape
        leader = x[0]

        # Downstream boundary relative to leader max position
        if L is None:
            L = np.nanmax(leader) - offset

        # Actual total time spent inside the boundary
        tts_actual = ((x < L).sum(dtype=float)) * dt

        # ---- check if last vehicle crossed boundary ----
        last_vehicle_pos = np.nanmax(x[-1])  # last vehicle max position
        if last_vehicle_pos < L:
            warnings.warn(
                f"Last vehicle did not cross the downstream boundary (max {last_vehicle_pos:.2f} < L={L:.2f}). "
                "Delay result may be invalid."
            )

        # Construct Newell benchmark trajectories (time + space shift)
        x_star = np.full_like(x, np.nan)
        for i in range(N):
            shift_t = int(round(i * tau / dt))
            shift_x = i * s
            if shift_t < T:
                leader_shifted = leader[:T - shift_t]
                x_star[i, shift_t:] = leader_shifted - shift_x

        # Benchmark total time spent
        tts_bench = ((x_star < L).sum(dtype=float)) * dt

        return float(tts_actual - tts_bench)


    @staticmethod
    def vt_micro_fleet_L_per_km(results: np.ndarray,
                                        dt: float,
                                        smooth_window_s: float = 0.5) -> float:
        """
        Compute fleet-average fuel consumption (L/km) after smoothing v and recomputing a.
        Uses TR-C 2013 Appendix A (eq. A.4) P_FC coefficients.

        Parameters
        ----------
        results : np.ndarray
            Shape (N, T, 3), columns [x, v, a], units: m, m/s, m/s^2
        dt : float
            Time step (s)
        smooth_window_s : float
            Smoothing window length (s). Default 0.5 s.

        Returns
        -------
        float
            Fleet-average L/km
        """

        def preprocess_v_a(x, v, dt, smooth_window_s=0.5):
            """
            Smooth velocity first, then recompute acceleration.
            """
            # Smoothing window
            win = max(1, int(round(smooth_window_s / dt)))
            
            # Smooth velocity to reduce noise amplification in acceleration
            v_s = gaussian_filter1d(v, sigma=win/2, axis=1, mode="nearest")
            
            # Recompute acceleration from smoothed velocity
            a_s = np.diff(v_s, axis=1, prepend=v_s[:, [0]]) / dt
            
            return v_s, a_s
        if results.ndim != 3 or results.shape[2] < 3:
            raise ValueError("results 必须是 (N, T, 3) 且包含 [x, v, a]")

        N, T, _ = results.shape
        x = results[..., 0]
        v = results[..., 1]
        a = results[..., 2]

        # Clamp to reasonable physical bounds before fuel model
        a = np.clip(a, -12, 6)
        v = np.clip(v, 0, 30)

        # --- Smooth v and recompute a ---
        v_s, a_s = preprocess_v_a(x, v, dt, smooth_window_s)

        # --- P_FC coefficient matrix (scaled by 0.01) ---
        K = 0.01 * np.array([
            [-753.7,   44.3809,  17.1641,  -4.2024],
            [   9.7326,  5.1753,   0.2942,  -0.7068],
            [  -0.3014, -0.0742,   0.0109,   0.0116],
            [   0.0053,  0.0006,  -0.0010,  -0.0006]
        ], dtype=float)

        # v^i, a^j
        v_pow = np.stack([v_s**i for i in range(4)], axis=-1)  # (N,T,4)
        a_pow = np.stack([a_s**j for j in range(4)], axis=-1)  # (N,T,4)

        # logF = Σ_i Σ_j K_ij v^i a^j
        logF = np.zeros((N, T))
        for i in range(4):
            for j in range(4):
                logF += K[i, j] * v_pow[..., i] * a_pow[..., j]


        # Prevent overflow
        logF = np.clip(logF, -50, 2)
        if np.any((logF > 2) | (logF < -50)):
            warnings.warn("logF 有值超出 [-50, 2] 范围", RuntimeWarning)
        F = np.exp(logF)  # L/s

        # Per-vehicle total fuel (L)
        per_vehicle_L = F.sum(axis=1) * dt
        # Per-vehicle distance (km) using delta-x
        per_vehicle_km = (x[:, -1] - x[:, 0]) / 1000.0

        # Fleet-average L/km
        with np.errstate(divide='ignore', invalid='ignore'):
            fleet_mean = float(np.nanmean(per_vehicle_L / per_vehicle_km))

        return fleet_mean
