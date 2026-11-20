"""Nonlinear rigid-body equations of motion for the Crazyflie 2.1, per Snider
Section 2 (p.2-3).

Forked from uav_sim/src/dynamics.py -- see params.py's docstring on why this
project is self-contained. One change from the uav_sim original: the inline
Euler-kinematics matrix build inside rigid_body_deriv is factored out into
euler_rate_matrix() below, so flatness.py can invert it exactly (desired
Euler-angle rates -> desired body rates) instead of re-deriving it.

State (12): [px, py, pz, u, v, w, phi, theta, psi, p, q, r]
    px, py, pz  -- position, earth frame [m]
    u, v, w     -- velocity, BODY frame [m/s]
    phi, theta, psi -- roll, pitch, yaw [rad], ZYX Euler
    p, q, r     -- body angular rates [rad/s]

Input (4): [Z, L, M, N]
    Z -- upward thrust [N]
    L, M, N -- roll, pitch, yaw moments [N*m]

theta = +-90 deg is a kinematic singularity (1/cos(theta) terms below) -- by
design this is not handled, only documented (matches uav_sim's own
tests/test_singularity.py).

Gravity is rotated into the body frame via the full rotation matrix (not the
report's own near-hover-only shortcut) -- see uav_sim/src/dynamics.py's
DEVIATION docstring and CHANGELOG.md for the full derivation; this fork
carries the same (already-corrected) physics forward unchanged.
"""
import numpy as np


def rotation_body_to_earth(phi: float, theta: float, psi: float) -> np.ndarray:
    """R = Rz(psi) @ Ry(theta) @ Rx(phi), body-frame vector -> earth frame."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    Rz = np.array([[cpsi, -spsi, 0.0], [spsi, cpsi, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[ctheta, 0.0, stheta], [0.0, 1.0, 0.0], [-stheta, 0.0, ctheta]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cphi, -sphi], [0.0, sphi, cphi]])
    return Rz @ Ry @ Rx


def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
    """E such that [phi_dot, theta_dot, psi_dot] = E @ [p, q, r] (ZYX Euler
    kinematics). Singular at theta = +-90deg (matches rotation_body_to_earth's
    own singularity -- both come from the same Euler-angle parameterization)."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, ttheta = np.cos(theta), np.tan(theta)
    return np.array([
        [1.0, sphi * ttheta, cphi * ttheta],
        [0.0, cphi, -sphi],
        [0.0, sphi / ctheta, cphi / ctheta],
    ])


def rigid_body_deriv(
    state12: np.ndarray,
    u4_actual: np.ndarray,
    params,
    gammas,
) -> np.ndarray:
    """dstate/dt for the 12-state rigid-body model. u4_actual is the *actual*
    (post motor-lag) [Z, L, M, N] -- see actuator.py / plant.py for the lag."""
    u, v, w = state12[3], state12[4], state12[5]
    phi, theta, psi = state12[6], state12[7], state12[8]
    p, q, r = state12[9], state12[10], state12[11]
    Z, L, M, N = u4_actual

    R = rotation_body_to_earth(phi, theta, psi)
    p_dot_earth = R @ np.array([u, v, w])

    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    uvw_dot = np.array([
        r * v - q * w + params.g * stheta,
        p * w - r * u - params.g * sphi * ctheta,
        q * u - p * v + Z / params.mass - params.g * cphi * ctheta,
    ])

    euler_dot = euler_rate_matrix(phi, theta) @ np.array([p, q, r])

    pqr_dot = np.array([
        gammas.g1 * p * q - gammas.g2 * q * r + gammas.g3 * L + gammas.g4 * N,
        gammas.g5 * p * r - gammas.g6 * (p ** 2 - r ** 2) + M / params.Jyy,
        gammas.g7 * p * q - gammas.g1 * q * r + gammas.g4 * L + gammas.g8 * N,
    ])

    return np.concatenate([p_dot_earth, uvw_dot, euler_dot, pqr_dot])
