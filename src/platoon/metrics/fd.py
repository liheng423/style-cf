import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor


class Parallelogram:
    """
    Parallelogram in time-space, centered at (t_center, x_center).

    Long edge is aligned with wave direction (1, wave_speed).
    Short edge is aligned with vehicle direction (1, given_speed).
    """

    def __init__(self, t_center, x_center, L, H, wave_speed, given_speed):
        self.t_center = t_center
        self.x_center = x_center
        self.L = L
        self.H = H
        self.wave_speed = wave_speed
        self.given_speed = given_speed
        self.corners = self._get_parallelogram_corners(t_center, x_center, L, H, wave_speed, given_speed)

    @staticmethod
    def _is_point_in_parallelogram(point, corners):
        """
        Check whether a 2D point lies within a parallelogram defined by corners.

        Parameters:
            point: array-like of shape (2,), e.g. [t, x]
            corners: ndarray of shape (4, 2)

        Returns:
            bool: True if point is inside the parallelogram
        """
        origin = corners[0]
        v1 = corners[3] - corners[0]  # long edge
        v2 = corners[1] - corners[0]  # short edge
        A = np.column_stack((v1, v2))
        b = point - origin
        try:
            uv = np.linalg.solve(A, b)
            u, v = uv
            return (0 <= u <= 1) and (0 <= v <= 1)
        except np.linalg.LinAlgError:
            return False

    @classmethod
    def from_proj(cls, t_center, x_center, L_t, H_x, wave_speed, given_speed):
        """
        Construct using axis projections.

        Parameters:
            L_t: float — projection of the long edge on the time axis
            H_x: float — projection of the short edge on the space axis
        """
        L = L_t * np.sqrt(1 + wave_speed**2)
        H = H_x * np.sqrt(1 + given_speed**2) / abs(given_speed) if given_speed != 0 else 0.0
        return cls(t_center, x_center, L, H, wave_speed, given_speed)

    @classmethod
    def from_proj_swapped(cls, t_center, x_center, H_t, L_x, wave_speed, given_speed):
        """
        Construct using swapped axis projections.

        Parameters:
            H_t: float — projection of the short edge on the time axis
            L_x: float — projection of the long edge on the space axis
        """
        H = H_t * np.sqrt(1 + given_speed**2)
        L = L_x * np.sqrt(1 + wave_speed**2) / abs(wave_speed) if wave_speed != 0 else 0.0
        return cls(t_center, x_center, L, H, wave_speed, given_speed)

    def contains(self, point):
        return self._is_point_in_parallelogram(point, self.corners)

    def plot(self, show_center=True, **kwargs):
        corners_closed = np.vstack([self.corners, self.corners[0]])
        plt.plot(corners_closed[:, 0], corners_closed[:, 1], **kwargs)
        if show_center:
            plt.plot(self.t_center, self.x_center, 'ko', label='center')
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.title("Parallelogram in Time-Space Domain")

    def area(self):
        # Use the same adjacent edges as contains() for consistency.
        v1 = self.corners[3] - self.corners[0]  # long edge
        v2 = self.corners[1] - self.corners[0]  # short edge
        return abs(np.linalg.det(np.column_stack((v1, v2))))

    @staticmethod
    def _get_parallelogram_corners(t_center, x_center, L, H, wave_speed, given_speed):
        """
        Build an Edie-style parallelogram.

        Long edge follows the wave direction (1, w).
        Short edge follows the vehicle direction (1, v_car).

        Parameters:
            t_center, x_center: float
                Center of the parallelogram (seconds, meters)
            L: float
                Length of the long edge (meters)
            H: float
                Length of the short edge (meters)
            wave_speed: float
                Wave speed (m/s)
            given_speed: float
                Vehicle speed (m/s)

        Returns:
            corners: ndarray of shape (4, 2), clockwise:
                bottom-left -> top-left -> top-right -> bottom-right
        """
        v_wave = np.array([1.0, wave_speed])
        v_wave = v_wave / np.linalg.norm(v_wave)  # unit vector, long-edge direction

        v_car = given_speed
        v_vehicle = np.array([1.0, v_car])
        v_vehicle = v_vehicle / np.linalg.norm(v_vehicle)  # unit vector, short-edge direction

        half_long = v_wave * (L / 2)
        half_short = v_vehicle * (H / 2)

        center = np.array([t_center, x_center])

        corner1 = center - half_long - half_short  # bottom-left
        corner2 = center - half_long + half_short  # top-left
        corner3 = center + half_long + half_short  # top-right
        corner4 = center + half_long - half_short  # bottom-right

        return np.vstack([corner1, corner2, corner3, corner4])


def compute_edie_qkv_parallelogram_matrix(car_trajs, dt_sample, parallelogram):
    """
    Compute Edie's q, k, v within a parallelogram.

    Parameters:
        car_trajs: ndarray of shape (N, T)
        dt_sample: float or ndarray of shape (T-1,)
        parallelogram: Parallelogram object

    Returns:
        q, k, v: float
    """
    N, T = car_trajs.shape

    if np.isscalar(dt_sample):
        dt_array = np.full(T - 1, dt_sample)
    else:
        dt_array = np.asarray(dt_sample)
        if dt_array.shape[0] != T - 1:
            raise ValueError("dt_sample must have shape (T-1,)")

    total_distance = 0.0
    total_time = 0.0

    for i in range(N):
        traj = car_trajs[i, :]
        for t in range(T - 1):
            x1, x2 = traj[t], traj[t + 1]
            if np.isnan(x1) or np.isnan(x2):
                continue

            dt = dt_array[t]
            t_mid = np.sum(dt_array[:t]) + 0.5 * dt
            x_mid = 0.5 * (x1 + x2)

            if parallelogram.contains(np.array([t_mid, x_mid])):
                total_distance += (x2 - x1)
                total_time += dt

    area = parallelogram.area()
    if area == 0 or total_time == 0:
        return 0.0, 0.0, 0.0

    q = total_distance / area
    k = total_time / area
    v = total_distance / total_time

    return q, k, v


def query_proj_from_intersections(
    results: np.ndarray,
    t_center: float,
    t_length: float,
    dt: float,
    wave_speed: float
):
    """
    From the center point, project along wave direction to the head and tail
    trajectories, then infer spatial propagation length from fixed H_t.

    Parameters:
        results: ndarray of shape (N, T, 3)
            Per-vehicle time series: [position, speed, acceleration]
        t_center: float
            Center time (s)
        t_length: float
            Fixed propagation time H_t (s)
        dt: float
            Time step (s)
        wave_speed: float
            Wave speed (m/s)

    Returns:
        t_center: float
        x_center: float
        H_t: float = t_length
        L_x: float
        given_speed: float
    """
    t_array = np.arange(results.shape[1]) * dt
    t_center_idx = int(round(t_center / dt))
    t_center = t_array[t_center_idx]

    x_head = results[0, :, 0]
    x_tail = results[-1, :, 0]

    x_head_c = x_head[t_center_idx]
    x_tail_c = x_tail[t_center_idx]

    if np.isnan(x_head_c) or np.isnan(x_tail_c):
        raise RuntimeError("Head or tail has NaN at center time.")

    x_center = 0.5 * (x_head_c + x_tail_c)
    center_point = np.array([t_center, x_center])

    # Wave-direction unit vector.
    wave_dir = np.array([1.0, wave_speed])
    wave_dir /= np.linalg.norm(wave_dir)

    # Mean vehicle speed (returned; sign does not affect geometry).
    v_head = results[0, :, 1]
    v_tail = results[-1, :, 1]
    given_speed = np.nanmean([np.nanmean(v_head), np.nanmean(v_tail)])

    def closest_point_projection(traj: np.ndarray, direction: np.ndarray):
        """
        Find the trajectory point with minimal perpendicular distance to
        the ray from center_point along the given direction.
        """
        valid = ~np.isnan(traj)
        traj_pts = np.stack([t_array[valid], traj[valid]], axis=1)
        rel_pts = traj_pts - center_point

        proj_len = rel_pts @ direction
        proj_vec = np.outer(proj_len, direction)
        perp_vec = rel_pts - proj_vec
        dists = np.linalg.norm(perp_vec, axis=1)

        closest = traj_pts[np.argmin(dists)]
        return closest

    # Find the two closest points along wave direction.
    head_on_wave = closest_point_projection(x_head, wave_dir)
    tail_on_wave = closest_point_projection(x_tail, wave_dir)

    L_x = abs(head_on_wave[1] - tail_on_wave[1])  # spatial propagation length

    x_center = 0.5 * (head_on_wave[1] + tail_on_wave[1])
    t_center = 0.5 * (head_on_wave[0] + tail_on_wave[0])

    return t_center, x_center, t_length, L_x, given_speed


def estimate_fd_params(density, flow):
    """
    Estimate traffic fundamental diagram parameters using robust linear fitting.

    Args:
        density (array-like): density values (veh/m)
        flow (array-like): flow values (veh/s)

    Returns:
        dict: {
            "capacity": float,
            "jam_density": float,
            "critical_density": float,
            "slope": float,
            "intercept": float
        }
    """
    density = np.array(density).reshape(-1, 1)
    flow = np.array(flow)

    # Robust linear fit.
    ransac = RANSACRegressor(
        base_estimator=LinearRegression(),
        min_samples=0.5,
        residual_threshold=0.01,
        random_state=42,
    )
    ransac.fit(density, flow)

    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    # Capacity = max observed flow.
    capacity = np.max(flow)

    # Jam density: flow = 0 -> density = -intercept / slope.
    jam_density = -intercept / slope if slope != 0 else np.nan

    # Critical density: where fitted flow equals capacity.
    critical_density = (capacity - intercept) / slope if slope != 0 else np.nan

    return {
        "capacity": capacity,
        "jam_density": jam_density,
        "critical_density": critical_density,
        "slope": slope,
        "intercept": intercept,
    }
