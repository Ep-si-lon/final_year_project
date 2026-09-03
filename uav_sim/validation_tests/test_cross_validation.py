"""Cross-validates this project's hand-written RK4 integrator against
scipy.integrate.solve_ivp integrating the *same* equations of motion
(augmented_deriv). This does not validate the equations themselves -- both
paths use identical dynamics -- but it is a genuine, independent check that
the hand-rolled RK4 step (rk4_step) correctly and accurately solves the ODE
it's supposed to, using a mature, independently-implemented high-accuracy
solver as ground truth. Catches integrator bugs: wrong step formula,
indexing errors, sign slips in how state is packed/unpacked, etc.
"""
import numpy as np
import pytest

from uav_sim.src.params import CrazyflieParams, compute_gammas
from uav_sim.src.plant import augmented_deriv, rk4_step, STATE_DIM, ACTUATOR_DIM

AUGMENTED_DIM = STATE_DIM + ACTUATOR_DIM


def test_rk4_matches_scipy_solve_ivp():
    from scipy.integrate import solve_ivp

    params = CrazyflieParams()
    gammas = compute_gammas(params)

    x0 = np.zeros(AUGMENTED_DIM)
    x0[7] = 0.1  # theta
    x0[9] = 0.5  # p
    x0[10] = -0.3  # q
    x0[11] = 0.2  # r
    x0[STATE_DIM:] = [params.mass * params.g, 0.0, 0.0, 0.0]  # settled hover lag

    u_cmd = np.array([params.mass * params.g * 1.05, 3e-4, -2e-4, 1e-4])

    T = 1.0
    dt = 0.001
    steps = int(T / dt)

    x = x0.copy()
    for _ in range(steps):
        x = rk4_step(augmented_deriv, x, u_cmd, dt, params, gammas)
    rk4_final = x

    def fun(t, x, u, params, gammas):
        return augmented_deriv(x, u, params, gammas)

    sol = solve_ivp(
        fun, [0, T], x0, args=(u_cmd, params, gammas),
        method="DOP853", rtol=1e-10, atol=1e-12, dense_output=False,
    )
    assert sol.success
    scipy_final = sol.y[:, -1]

    assert np.all(np.isfinite(rk4_final))
    assert np.all(np.isfinite(scipy_final))
    assert np.allclose(rk4_final, scipy_final, atol=1e-4, rtol=1e-4)


def test_gym_pybullet_drones_cross_validation():
    """Placeholder for a real cross-validation against an independently
    implemented quadrotor simulator (gym-pybullet-drones/PyBullet), which
    would validate the equations of motion themselves rather than just the
    integrator. Not run here: it's a heavy external dependency (separate
    clone + patched URDF, PyBullet physics engine) out of scope for this
    sandbox. Before a fair comparison were possible, cf2x.urdf's mass
    (0.033kg vs the library's default ~0.027kg) and inertia (Jxx=16.57e-6
    etc vs the library's default ~1.4e-5) would need to be patched to match
    this project's Eq.4 values.
    """
    try:
        import gym_pybullet_drones  # noqa: F401
    except ImportError:
        pytest.skip(
            "gym-pybullet-drones not installed; see docstring for how a real "
            "comparison would need to patch cf2x.urdf's mass (0.033kg vs the "
            "library's default ~0.027kg) and inertia (Jxx=16.57e-6 etc vs the "
            "library's default ~1.4e-5) to match this project's Eq.4 values "
            "before a fair comparison is possible"
        )
