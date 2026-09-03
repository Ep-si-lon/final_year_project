"""Augmented plant: rigid-body dynamics (dynamics.py) + motor lag (actuator.py)
integrated together via RK4, exposed as a gym-like CrazyflieSim.

Deviation from the report, disclosed: Snider's own MATLAB script (Listing 4,
p.21-22) integrates with simple forward Euler at dt=1/500s. This module uses
RK4 at the same default dt instead -- strictly better-conditioned for a
reusable simulator, and not a silent "fix" of the report's math.
"""
from typing import Callable, Optional

import numpy as np

from .actuator import lag_deriv
from .dynamics import rigid_body_deriv
from .params import CrazyflieParams, Gammas, compute_gammas

STATE_DIM = 12
ACTUATOR_DIM = 4
AUGMENTED_DIM = STATE_DIM + ACTUATOR_DIM


def augmented_deriv(
    x16: np.ndarray,
    u_cmd4: np.ndarray,
    params: CrazyflieParams,
    gammas: Gammas,
) -> np.ndarray:
    """dstate/dt for the 16-state [rigid_body(12), actuator_lag(4)] system."""
    state12, lag4 = x16[:STATE_DIM], x16[STATE_DIM:]
    return np.concatenate([
        rigid_body_deriv(state12, lag4, params, gammas),
        lag_deriv(lag4, u_cmd4, params.tau_motor),
    ])


def rk4_step(
    f: Callable[..., np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    *args,
) -> np.ndarray:
    """Classic 4th-order Runge-Kutta step for dx/dt = f(x, u, *args)."""
    k1 = f(x, u, *args)
    k2 = f(x + 0.5 * dt * k1, u, *args)
    k3 = f(x + 0.5 * dt * k2, u, *args)
    k4 = f(x + dt * k3, u, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class CrazyflieSim:
    """Gym-like plant simulator: instantiate, then call .step(u_cmd, dt)
    repeatedly with commanded [Z, L, M, N]. Build a controller around this."""

    def __init__(
        self,
        params: Optional[CrazyflieParams] = None,
        x0: Optional[np.ndarray] = None,
    ):
        self.params = params if params is not None else CrazyflieParams()
        self.gammas = compute_gammas(self.params)
        if x0 is None:
            x0 = np.zeros(STATE_DIM)
        elif x0.shape != (STATE_DIM,):
            raise ValueError(f"x0 must have shape ({STATE_DIM},), got {x0.shape}")
        # Actuator lag states start settled at hover thrust, zero moments.
        lag0 = np.array([self.params.mass * self.params.g, 0.0, 0.0, 0.0])
        self._x16 = np.concatenate([x0, lag0])

    def step(self, u_cmd: np.ndarray, dt: float = 0.002) -> np.ndarray:
        """Advance the plant by dt under commanded [Z, L, M, N]. Returns the
        new 12-vector rigid-body state."""
        u_cmd = np.asarray(u_cmd, dtype=float)
        if u_cmd.shape != (ACTUATOR_DIM,):
            raise ValueError(f"u_cmd must have shape ({ACTUATOR_DIM},), got {u_cmd.shape}")
        self._x16 = rk4_step(augmented_deriv, self._x16, u_cmd, dt, self.params, self.gammas)
        return self.state

    @property
    def state(self) -> np.ndarray:
        return self._x16[:STATE_DIM].copy()

    @property
    def actuator_state(self) -> np.ndarray:
        return self._x16[STATE_DIM:].copy()
