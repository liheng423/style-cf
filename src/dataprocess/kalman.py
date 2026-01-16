import numpy as np

def kf(veh: np.ndarray, params: dict) -> np.ndarray:
    """
    Kalman Filter for 1D motion in y-direction with position and velocity measurements.

    Args:
        veh: np.ndarray of shape (T, 4), each row is [x, y, v, t], but only y, v, t are used.
        params: Dictionary containing:
            'uncertainty_init', 'uncertainty_pos', 'uncertainty_speed'

    Returns:
        estimates: np.ndarray of shape (T, 2), each row is [y_kf, v_kf]
    """
    uncertainty_init = params['uncertainty_init']
    uncertainty_pos = params['uncertainty_pos']
    uncertainty_speed = params['uncertainty_speed']
    max_acc = params['max_acc']
    # Extract measurements
    y_meas = veh[:, 1]
    v_meas = veh[:, 2]
    t = veh[:, 3]


    # State: [y, v]
    n = len(y_meas)
    x = np.array([y_meas[0], v_meas[0]])  # initial state
    P = np.eye(2) * uncertainty_init      # initial covariance

    # State transition matrix A and measurement matrix H
    estimates = np.zeros((n, 2))
    estimates[0] = x
    I = np.eye(2)

    R = np.diag([uncertainty_pos, uncertainty_speed])  # Measurement noise

    for k in range(1, n):
        dT = t[k] - t[k-1]
        A = np.array([[1, dT], [0, 1]])
        H = np.eye(2)
        s_pos = 0.5 * max_acc * dT ** 2
        s_speed = max_acc * dT
        Q = np.diag([s_pos**2, s_speed**2])

        # Prediction
        x = A @ x
        P = A @ P @ A.T + Q

        # Update
        z = np.array([y_meas[k], v_meas[k]])
        y_residual = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y_residual
        P = (I - K @ H) @ P

        estimates[k, :] = x

    return estimates
