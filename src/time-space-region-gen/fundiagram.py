import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression

class Shape:
    
    def contains(self, point):
        pass

    def area(self):
        pass


class Rectangle(Shape):
    pass
    
class Parallelogram(Shape):
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
        构造函数：使用轴向投影值创建 Parallelogram 对象。

        Parameters:
            L_t: float — 长边在时间轴的投影
            H_x: float — 短边在空间轴的投影
        """
        L = L_t * np.sqrt(1 + wave_speed**2)
        H = H_x * np.sqrt(1 + given_speed**2) / abs(given_speed) if given_speed != 0 else 0.0
        return cls(t_center, x_center, L, H, wave_speed, given_speed)
    
    @classmethod
    def from_proj_swapped(cls, t_center, x_center, H_t, L_x, wave_speed, given_speed):
        """
        构造函数：使用轴向投影值创建 Parallelogram 对象（输入的是 H_t: 短边在时间轴投影，L_x: 长边在空间轴投影）
        """
        H = H_t * np.sqrt(1 + given_speed**2)
        L = L_x * np.sqrt(1 + wave_speed**2) / abs(wave_speed) if wave_speed != 0 else 0.0
        return cls(t_center, x_center, L, H, wave_speed, given_speed)

    def contains(self, point):
        return self._is_point_in_parallelogram(point, self.corners)

    def area(self):
        # 取与 contains() 一致的两条相邻边定义
        v1 = self.corners[3] - self.corners[0]  # long edge
        v2 = self.corners[1] - self.corners[0]  # short edge
        return abs(np.linalg.det(np.column_stack((v1, v2))))


    @staticmethod
    def _get_parallelogram_corners(t_center, x_center, L, H, wave_speed, given_speed):
        """
        构造 Edie-style parallelogram:
        - 长边沿波速方向 (1, w)
        - 短边沿车辆行驶方向 (1, v_car)

        Parameters:
            t_center, x_center: float
                平行四边形中心点 (秒, 米)
            L: float
                长边长度 (meters)
            H: float
                短边长度 (meters)
            wave_speed: float
                波速 (m/s)
            given_speed: float
                车辆速度 (m/s)

        Returns:
            corners: ndarray of shape (4, 2), 顺时针角点：左下 → 左上 → 右上 → 右下
        """
        v_wave = np.array([1.0, wave_speed])
        v_wave = v_wave / np.linalg.norm(v_wave)   # 单位向量：长边方向

        v_car = given_speed 
        v_vehicle = np.array([1.0, v_car])
        v_vehicle = v_vehicle / np.linalg.norm(v_vehicle)  # 单位向量：短边方向

        half_long = v_wave * (L / 2)
        half_short = v_vehicle * (H / 2)

        center = np.array([t_center, x_center])

        corner1 = center - half_long - half_short  # 左下
        corner2 = center - half_long + half_short  # 左上
        corner3 = center + half_long + half_short  # 右上
        corner4 = center + half_long - half_short  # 右下

        return np.vstack([corner1, corner2, corner3, corner4])


def compute_edie_qkv_parallelogram_matrix(car_trajs, dt_sample, region):
    """
    Compute Edie's q, k, v inside a parallelogram object.

    Parameters:
        car_trajs: ndarray of shape (N, T)
        dt_sample: float or ndarray of shape (T-1,)
        region (Shape): could be parallelogram, rectangle or any other closed region
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

            if region.contains(np.array([t_mid, x_mid])):
                total_distance += (x2 - x1)
                total_time += dt

    area = region.area()
    if area == 0 or total_time == 0:
        return 0.0, 0.0, 0.0

    q = total_distance / area
    k = total_time / area
    v = total_distance / total_time

    return q, k, v


def from_time_center(
    results: np.ndarray,
    t_center: float,
    t_length: float,
    dt: float,
    wave_speed: float
):
    """
    从中心点出发，沿波速方向查找与 head 和 tail 轨迹的交点，
    用用户输入的固定传播时间 H_t 推算空间传播长度 L_x。

    Parameters:
        results: ndarray of shape (N, T, 3)
            每辆车的 [位置, 速度, 加速度] 随时间变化的数组
        t_center: float
            中心时间（秒）
        t_length: float
            固定传播时间 H_t（秒）
        dt: float
            时间步长（秒）
        wave_speed: float
            波速（m/s）

    Returns:
        parallelogram: Paralellogram
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

    # 波速方向向量（单位向量）
    wave_dir = np.array([1.0, wave_speed])
    wave_dir /= np.linalg.norm(wave_dir)

    # 平均车速（用于返回，方向不影响）
    v_head = results[0, :, 1]
    v_tail = results[-1, :, 1]
    given_speed = np.nanmean([np.nanmean(v_head), np.nanmean(v_tail)])

    def closest_point_projection(traj: np.ndarray, direction: np.ndarray):
        """
        查找整个轨迹中距离方向直线（从 center_point 出发）垂直距离最小的点。
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

    # 在整条轨迹中找 wave 方向最近的两个交点
    head_on_wave = closest_point_projection(x_head, wave_dir)
    tail_on_wave = closest_point_projection(x_tail, wave_dir)

    L_x = abs(head_on_wave[1] - tail_on_wave[1])  # 空间传播距离


    x_center = 0.5 * (head_on_wave[1] + tail_on_wave[1])
    t_center = 0.5 * (head_on_wave[0] + tail_on_wave[0])

    return Parallelogram.from_proj_swapped(t_center, x_center, t_length, L_x, wave_speed, given_speed)



