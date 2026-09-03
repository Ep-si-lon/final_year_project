"""Validates the dynamics/actuator modules against closed-form, analytically
known solutions (free fall, exponential motor lag, trim-control approximation
from Snider Eq. 9), independent of the CrazyflieSim/motor-lag integration
tested elsewhere.
"""
import numpy as np

from uav_sim.src.params import CrazyflieParams, compute_gammas
from uav_sim.src.dynamics import rigid_body_deriv
from uav_sim.src.actuator import lag_deriv


def _rk4_step(deriv_fn, state, dt):
    k1 = deriv_fn(state)
    k2 = deriv_fn(state + 0.5 * dt * k1)
    k3 = deriv_fn(state + 0.5 * dt * k2)
    k4 = deriv_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def test_free_fall_matches_exact_parabola():
    """Zero thrust/torque from rest: phi=theta=psi=p=q=r stay exactly zero
    (no torque ever acts on them), R stays identity, so w and pz reduce to
    exact free fall: w(t) = -g*t, pz(t) = -0.5*g*t**2."""
    params = CrazyflieParams()
    gammas = compute_gammas(params)
    u4 = np.zeros(4)

    state = np.zeros(12)
    dt = 0.001
    steps = 1000

    deriv_fn = lambda s: rigid_body_deriv(s, u4, params, gammas)
    for _ in range(steps):
        state = _rk4_step(deriv_fn, state, dt)

    t = steps * dt
    pz_exact = -0.5 * params.g * t ** 2
    w_exact = -params.g * t

    assert np.isclose(state[2], pz_exact, atol=1e-9)  # pz
    assert np.isclose(state[5], w_exact, atol=1e-9)  # w
    zero_idx = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11]  # px,py,u,v,phi,theta,psi,p,q,r
    assert np.allclose(state[zero_idx], 0.0, atol=1e-9)


def test_motor_lag_matches_exact_exponential():
    """First-order lag from rest under a constant step command has the exact
    closed-form solution actual(t) = cmd*(1 - exp(-t/tau))."""
    tau = 0.1
    u4_cmd = np.array([1.0, 0.001, 0.001, 0.001])

    lag_state = np.zeros(4)
    dt = 0.0005
    steps = 1000  # 0.5s

    deriv_fn = lambda s: lag_deriv(s, u4_cmd, tau)
    for i in range(steps):
        lag_state = _rk4_step(deriv_fn, lag_state, dt)
        t = (i + 1) * dt
        expected = u4_cmd * (1.0 - np.exp(-t / tau))
        assert np.allclose(lag_state, expected, atol=1e-6)


def test_constant_roll_torque_matches_trim_approximation():
    """Hover thrust + small roll moment at rest: report's trim approximation
    (Eq. 9) says p_dot = L/Jxx exactly, but the exact gamma-based dynamics
    include a Jxz cross-inertia term that makes g3 differ from 1/Jxx by
    ~0.1%, hence the 0.5% relative tolerance rather than exact equality.

    rdot is NOT expected to be zero here: with p=q=r=N=0, rdot reduces to
    gammas.g4*L, a genuine nonzero yaw-rate side effect of a pure roll
    moment caused by the Jxz product-of-inertia coupling -- not a bug.
    """
    params = CrazyflieParams()
    gammas = compute_gammas(params)
    L = 1e-5
    u4 = np.array([params.mass * params.g, L, 0.0, 0.0])

    deriv = rigid_body_deriv(np.zeros(12), u4, params, gammas)

    pdot, qdot, rdot = deriv[9], deriv[10], deriv[11]

    pdot_trim = L / params.Jxx
    rel_err = abs(pdot - pdot_trim) / abs(pdot_trim)
    assert rel_err < 0.005

    assert np.isclose(qdot, 0.0, atol=1e-12)

    rdot_expected = gammas.g4 * L
    assert np.isclose(rdot, rdot_expected, rtol=1e-6)
