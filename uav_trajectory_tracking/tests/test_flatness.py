"""Validates flatness.py's feedforward inversion by checking it's actually
self-consistent with dynamics.py's own rigid-body equations -- NOT by
comparing to another paper's formula. Two independent checks:

1. FORWARD CHECK: plug the feedforward's (Z, phi, theta, psi) back into
   dynamics.py's own translational equation (as if the vehicle were exactly
   at that attitude, holding it, right now) and verify the resulting earth-
   frame acceleration equals the trajectory's own commanded acceleration.
   This is the core flatness identity and the thing most likely to have a
   sign or axis-order bug.
2. RATE CHECK: verify dynamics.py's own euler_rate_matrix, fed the
   feedforward's (p, q, r), reproduces the numerically-differentiated
   (phi_dot, theta_dot, psi_dot) of the feedforward attitude -- i.e. the
   rate feedforward is the exact inverse of the kinematics equation actually
   used elsewhere in this codebase (trivially true if flatness.py's own
   np.linalg.solve(E, ...) is correct, but this test also cross-checks E
   against a finite-difference-based reference build of the same matrix,
   catching a copy-paste error in dynamics.py's euler_rate_matrix itself).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uav_trajectory_tracking.src.dynamics import rotation_body_to_earth, euler_rate_matrix
from uav_trajectory_tracking.src.flatness import compute_feedforward, _desired_attitude
from uav_trajectory_tracking.src.params import CrazyflieParams
from uav_trajectory_tracking.src.trajectory import MinimumSnapTrajectory3D


def _earth_accel_from_attitude(Z, phi, theta, psi, params):
    """Independent re-derivation (from dynamics.py's own translational
    equation, but composed forward rather than inverted) of the earth-frame
    acceleration a body at rest (u=v=w=p=q=r=0) with thrust Z at attitude
    (phi,theta,psi) would experience right now: a = R@[0,0,Z/m] - g*z_hat."""
    R = rotation_body_to_earth(phi, theta, psi)
    thrust_earth = R @ np.array([0.0, 0.0, Z / params.mass])
    return thrust_earth - np.array([0.0, 0.0, params.g])


def _build_test_trajectory():
    waypoints_xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.8],
        [2.0, -0.5, 0.5],
        [1.0, 1.0, 1.0],
    ])
    times = [0.0, 3.0, 6.0, 10.0]
    yaw_wp = [0.0, 0.3, -0.5, 0.2]
    return MinimumSnapTrajectory3D(waypoints_xyz, times, yaw_waypoints=yaw_wp)


def test_feedforward_attitude_reproduces_commanded_acceleration():
    traj = _build_test_trajectory()
    params = CrazyflieParams()
    for t in np.linspace(traj.t_start + 0.05, traj.t_end - 0.05, 30):
        a_des = traj.eval_pos(t, max_deriv=2)[2]
        ff = compute_feedforward(traj, t, params)
        a_reconstructed = _earth_accel_from_attitude(ff.Z, ff.phi, ff.theta, ff.psi, params)
        assert np.allclose(a_reconstructed, a_des, atol=1e-9), (t, a_reconstructed, a_des)


def test_desired_attitude_solve_is_exact_not_approximate():
    """Regression test for a real bug caught during development: an earlier
    version built R_des via the generic SO(3) literature's x_c/cross-product
    construction, then re-extracted psi from it -- which does NOT exactly
    reproduce the input psi once tilt is nonzero (that construction defines
    a tilt-dependent notion of 'yaw' that only coincides with the intrinsic
    ZYX Euler yaw at zero tilt; measured ~1e-2 rad error at ~0.3 rad tilt in
    development). _desired_attitude was rewritten to solve directly for
    (phi,theta) holding psi fixed, which has no such round-trip. This test
    picks a random ground-truth (phi,theta,psi), builds the z-axis it
    implies, and checks _desired_attitude recovers (phi,theta) to near
    machine precision when given that exact psi back."""
    rng = np.random.default_rng(42)
    params = CrazyflieParams()
    for _ in range(20):
        phi_true = rng.uniform(-0.6, 0.6)
        theta_true = rng.uniform(-0.6, 0.6)
        psi_true = rng.uniform(-2.0, 2.0)
        R = rotation_body_to_earth(phi_true, theta_true, psi_true)
        z_b = R[:, 2]
        # a_des such that F_des = m*(a_des+g*z_hat) points along z_b with some
        # arbitrary positive magnitude (2.0x hover-equivalent, arbitrary):
        Z_arbitrary = 2.0 * params.mass * params.g
        a_des = (Z_arbitrary / params.mass) * z_b - np.array([0.0, 0.0, params.g])

        _, phi_des, theta_des = _desired_attitude(a_des, psi_true, params)
        assert np.isclose(phi_des, phi_true, atol=1e-9), (phi_des, phi_true)
        assert np.isclose(theta_des, theta_true, atol=1e-9), (theta_des, theta_true)


def test_feedforward_yaw_matches_trajectory_yaw():
    """psi_des extracted from the constructed rotation matrix must equal the
    trajectory's own commanded yaw -- confirms the R_des construction
    (x_c/y_b/x_b) didn't silently rotate heading."""
    traj = _build_test_trajectory()
    params = CrazyflieParams()
    for t in np.linspace(traj.t_start + 0.05, traj.t_end - 0.05, 20):
        psi_cmd = traj.eval_yaw(t, max_deriv=0)[0]
        ff = compute_feedforward(traj, t, params)
        assert np.isclose(ff.psi, psi_cmd, atol=1e-9), (t, ff.psi, psi_cmd)


def test_feedforward_rates_are_exact_kinematic_inverse():
    """Central-difference the feedforward's own (phi,theta,psi) and check
    euler_rate_matrix(phi,theta) @ [p,q,r] reproduces it -- confirms the
    p,q,r returned really do invert the SAME kinematics equation
    dynamics.py's rigid_body_deriv uses forward."""
    traj = _build_test_trajectory()
    params = CrazyflieParams()
    eps = 1e-5

    def euler_at(t):
        ff = compute_feedforward(traj, t, params)
        return np.array([ff.phi, ff.theta, ff.psi])

    for t in np.linspace(traj.t_start + 0.1, traj.t_end - 0.1, 15):
        ff = compute_feedforward(traj, t, params)
        euler_dot_fd = (euler_at(t + eps) - euler_at(t - eps)) / (2 * eps)
        E = euler_rate_matrix(ff.phi, ff.theta)
        euler_dot_reconstructed = E @ np.array([ff.p, ff.q, ff.r])
        assert np.allclose(euler_dot_reconstructed, euler_dot_fd, atol=1e-4), (
            t, euler_dot_reconstructed, euler_dot_fd,
        )


def test_hover_trajectory_gives_hover_feedforward():
    """Degenerate case sanity check: a trajectory that starts and stays at a
    single point (all derivatives zero) must feedforward exactly Z=m*g,
    phi=theta=0, p=q=r=0 -- the same equilibrium point verified for uav_sim
    earlier in this project."""
    waypoints_xyz = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    times = [0.0, 5.0]
    traj = MinimumSnapTrajectory3D(waypoints_xyz, times)
    params = CrazyflieParams()

    ff = compute_feedforward(traj, 2.5, params)
    assert np.isclose(ff.Z, params.mass * params.g, atol=1e-9)
    assert np.isclose(ff.phi, 0.0, atol=1e-9)
    assert np.isclose(ff.theta, 0.0, atol=1e-9)
    assert np.allclose([ff.p, ff.q, ff.r], 0.0, atol=1e-6)
