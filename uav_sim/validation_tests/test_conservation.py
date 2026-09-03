"""Torque-free rigid-body rotation must exactly conserve rotational kinetic
energy and angular-momentum magnitude (Euler's equations for L=M=N=0). This
is a strong correctness check on the gamma-based pdot/qdot/rdot equations in
dynamics.rigid_body_deriv -- any sign error or typo there will leak or gain
energy/momentum over the rollout.

The gamma formulas are the exact solution of J_eff @ omegadot =
torque - omega x (J_eff @ omega) for the effective inertia tensor with a
MINUS sign on Jxz off-diagonal (standard aerospace product-of-inertia
convention):
    J_eff = [[Jxx, 0, -Jxz], [0, Jyy, 0], [-Jxz, 0, Jzz]]
"""
import numpy as np

from uav_sim.src.dynamics import rigid_body_deriv
from uav_sim.src.params import CrazyflieParams, compute_gammas


def _rk4_step(state12, u4, params, gammas, dt):
    k1 = rigid_body_deriv(state12, u4, params, gammas)
    k2 = rigid_body_deriv(state12 + 0.5 * dt * k1, u4, params, gammas)
    k3 = rigid_body_deriv(state12 + 0.5 * dt * k2, u4, params, gammas)
    k4 = rigid_body_deriv(state12 + dt * k3, u4, params, gammas)
    return state12 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _energy_and_momentum(p, q, r, params):
    """Rotational kinetic energy T and body-frame angular momentum H for the
    effective inertia tensor (note the -Jxz off-diagonal, see module docstring)."""
    T = 0.5 * (params.Jxx * p ** 2 + params.Jyy * q ** 2 + params.Jzz * r ** 2
               - 2 * params.Jxz * p * r)
    H = np.array([
        params.Jxx * p - params.Jxz * r,
        params.Jyy * q,
        -params.Jxz * p + params.Jzz * r,
    ])
    return T, H


def _run_torque_free_and_check_conservation(p0, q0, r0):
    params = CrazyflieParams()
    gammas = compute_gammas(params)
    u4 = np.array([params.mass * params.g, 0.0, 0.0, 0.0])  # torque-free: L=M=N=0

    state = np.zeros(12)
    state[9], state[10], state[11] = p0, q0, r0

    dt = 0.0005
    steps = int(2.0 / dt)  # 2s

    T0, H0 = _energy_and_momentum(p0, q0, r0, params)
    Hmag0 = np.linalg.norm(H0)

    max_T_rel_err = 0.0
    max_H_rel_err = 0.0
    for _ in range(steps):
        state = _rk4_step(state, u4, params, gammas, dt)
        p, q, r = state[9], state[10], state[11]
        T, H = _energy_and_momentum(p, q, r, params)
        max_T_rel_err = max(max_T_rel_err, abs(T - T0) / abs(T0))
        max_H_rel_err = max(max_H_rel_err, abs(np.linalg.norm(H) - Hmag0) / Hmag0)

    return max_T_rel_err, max_H_rel_err


def test_torque_free_rotation_conserves_energy_and_momentum():
    max_T_rel_err, max_H_rel_err = _run_torque_free_and_check_conservation(2.0, -1.5, 3.0)
    assert max_T_rel_err < 1e-8
    assert max_H_rel_err < 1e-8


def test_conservation_holds_for_a_different_initial_spin():
    max_T_rel_err, max_H_rel_err = _run_torque_free_and_check_conservation(-3.0, 4.0, -1.0)
    assert max_T_rel_err < 1e-8
    assert max_H_rel_err < 1e-8
