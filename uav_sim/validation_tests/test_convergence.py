"""Self-convergence (Richardson doubling) check that plant.rk4_step actually
integrates at 4th order, not just something that looks like RK4 in the code.

Integrates the 12-state rigid-body dynamics from a fixed nonlinear initial
state under a fixed off-hover control at three step sizes h, h/2, h/4 and
checks that the error halves by ~16x each time step size halves -- the
signature of a true 4th-order method (2nd order would give ~4x, 1st order ~2x).
"""
import numpy as np

from uav_sim.src.params import CrazyflieParams, compute_gammas
from uav_sim.src.dynamics import rigid_body_deriv
from uav_sim.src.plant import rk4_step


def _integrate(state0, u4, params, gammas, T, h):
    state = state0.copy()
    steps = int(round(T / h))
    for _ in range(steps):
        state = rk4_step(rigid_body_deriv, state, u4, h, params, gammas)
    return state


def test_rk4_achieves_fourth_order_convergence():
    params = CrazyflieParams()
    gammas = compute_gammas(params)

    state0 = np.array([0, 0, 0, 0.5, 0.3, 0.1, 0.1, 0.15, 0.05, 0.8, -0.5, 1.2])
    u4 = np.array([params.mass * params.g * 1.1, 2e-4, -1.5e-4, 1e-4])

    T = 0.5
    h = 0.01

    x_h = _integrate(state0, u4, params, gammas, T, h)
    x_h2 = _integrate(state0, u4, params, gammas, T, h / 2)
    x_h4 = _integrate(state0, u4, params, gammas, T, h / 4)

    e1 = np.linalg.norm(x_h - x_h2)
    e2 = np.linalg.norm(x_h2 - x_h4)

    # Observed during development: e1≈1.322e-07, e2≈8.244e-09, ratio≈16.04
    assert e2 > 1e-14
    ratio = e1 / e2
    assert 8 <= ratio <= 24
    assert np.all(np.isfinite(x_h4))
