"""Nonlinear rigid-body equations of motion for the Crazyflie 2.1, per Snider
Section 2 (p.2-3).

State (12): [px, py, pz, u, v, w, phi, theta, psi, p, q, r]
    px, py, pz  -- position, earth frame [m]
    u, v, w     -- velocity, BODY frame [m/s]
    phi, theta, psi -- roll, pitch, yaw [rad], ZYX Euler
    p, q, r     -- body angular rates [rad/s]

Input (4): [Z, L, M, N]
    Z -- upward thrust [N]
    L, M, N -- roll, pitch, yaw moments [N*m]

theta = +-90 deg is a kinematic singularity (1/cos(theta) terms below) -- by
design this is not handled, only documented (see tests/test_singularity.py).
"""
import numpy as np

from .params import CrazyflieParams, Gammas


def rotation_body_to_earth(phi: float, theta: float, psi: float) -> np.ndarray:
    """R = Rz(psi) @ Ry(theta) @ Rx(phi), body-frame vector -> earth frame."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    Rz = np.array([[cpsi, -spsi, 0.0], [spsi, cpsi, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[ctheta, 0.0, stheta], [0.0, 1.0, 0.0], [-stheta, 0.0, ctheta]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cphi, -sphi], [0.0, sphi, cphi]])
    return Rz @ Ry @ Rx


def rigid_body_deriv(
    state12: np.ndarray,
    u4_actual: np.ndarray,
    params: CrazyflieParams,
    gammas: Gammas,
) -> np.ndarray:
    """dstate/dt for the 12-state rigid-body model. u4_actual is the *actual*
    (post motor-lag) [Z, L, M, N] -- see actuator.py / plant.py for the lag."""
    u, v, w = state12[3], state12[4], state12[5]
    phi, theta, psi = state12[6], state12[7], state12[8]
    p, q, r = state12[9], state12[10], state12[11]
    Z, L, M, N = u4_actual

    R = rotation_body_to_earth(phi, theta, psi)
    p_dot_earth = R @ np.array([u, v, w])

    uvw_dot = np.array([r * v - q * w, p * w - r * u, q * u - p * v])
    uvw_dot[2] += (Z - params.mass * params.g) / params.mass

    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, ttheta = np.cos(theta), np.tan(theta)
    euler_kinematics = np.array([
        [1.0, sphi * ttheta, cphi * ttheta],
        [0.0, cphi, -sphi],
        [0.0, sphi / ctheta, cphi / ctheta],
    ])
    euler_dot = euler_kinematics @ np.array([p, q, r])

    pqr_dot = np.array([
        gammas.g1 * p * q - gammas.g2 * q * r + gammas.g3 * L + gammas.g4 * N,
        gammas.g5 * p * r - gammas.g6 * (p ** 2 - r ** 2) + M / params.Jyy,
        gammas.g7 * p * q - gammas.g1 * q * r + gammas.g4 * L + gammas.g8 * N,
    ])

    return np.concatenate([p_dot_earth, uvw_dot, euler_dot, pqr_dot])
