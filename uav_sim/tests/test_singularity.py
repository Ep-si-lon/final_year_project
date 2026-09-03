"""theta = +-90 deg is a kinematic singularity of the Euler-angle rate
equations (1/cos(theta) terms). This is documented, not handled -- the plant
is not expected to be usable through gimbal lock."""
import numpy as np

from uav_sim.src.dynamics import rigid_body_deriv
from uav_sim.src.params import CrazyflieParams, compute_gammas


def test_theta_90deg_blows_up():
    params = CrazyflieParams()
    gammas = compute_gammas(params)

    state = np.zeros(12)
    state[7] = np.pi / 2  # theta
    state[11] = 1.0  # r, nonzero rate to excite the singular cphi/ctheta term (phi=0 => cphi=1)
    u = np.array([params.mass * params.g, 0.0, 0.0, 0.0])

    xdot = rigid_body_deriv(state, u, params, gammas)
    psi_dot = xdot[8]

    assert not np.isfinite(psi_dot) or abs(psi_dot) > 1e10
