"""Mixer: maps commanded [Z, L, M, N] to per-motor squared speeds, per Snider
Eq. 20/22 (p.24-25). Reference-only -- NOT wired into CrazyflieSim by default,
since the report's own Section 7 closed-loop simulation feeds [Z, L, M, N]
directly into the rigid-body dynamics without going through the mixer; the
mixer is a firmware/hardware detail (Section 9), not part of the plant.

Note: the report's own firmware C listing (Listing 5, p.26) uses a *third*,
mutually-inconsistent sign pattern for mixer_inv that matches neither Eq. 22
nor a consistent matrix inverse of Eq. 20 below. Eq. 22 (as printed in the
report body, not the firmware appendix) is treated here as canonical.
"""
import numpy as np

MIXER = np.array([
    [1.0, 1.0, 1.0, 1.0],
    [-1.0, -1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0, -1.0],
])

MIXER_INV = 0.25 * np.array([
    [1.0, -1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0, -1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0, -1.0],
])


def forces_to_motor_speeds_squared(zlmn: np.ndarray) -> np.ndarray:
    """[Z, L, M, N] -> [n1^2, n2^2, n3^2, n4^2], up to the rho*C_T*D^5 scale
    factor the report folds into its own empirical PWM<->thrust curve (Eq. 3)
    instead of giving separately (Eq. 21)."""
    return MIXER_INV @ np.asarray(zlmn, dtype=float)
