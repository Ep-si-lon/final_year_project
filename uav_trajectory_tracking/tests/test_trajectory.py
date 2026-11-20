"""Validates trajectory.py's minimum-snap solver against:
1. A known closed-form point-to-point minimum-snap polynomial (independent
   derivation, not from this module) -- catches sign/index errors in the QP.
2. Internal continuity (velocity/accel/jerk match across segment boundaries)
   for a genuinely multi-segment (3+ waypoint) case.
3. Waypoints hit exactly at their specified times.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uav_trajectory_tracking.src.trajectory import MinimumSnapTrajectory3D, min_snap_1d


def closed_form_min_snap_point_to_point(x0, x1, T, tau):
    """Independent closed-form for the r=4 (snap) point-to-point trajectory
    with zero velocity/accel/jerk at both ends -- the degree-7 'smootherstep'
    polynomial family. s = tau/T in [0,1]:
        p(s) = x0 + (x1-x0)*(35 s^4 - 84 s^5 + 70 s^6 - 20 s^7)
    This is a standard, independently-known result (not derived from this
    module's QP code), used here purely as an oracle.
    """
    s = tau / T
    return x0 + (x1 - x0) * (35 * s**4 - 84 * s**5 + 70 * s**6 - 20 * s**7)


def test_single_segment_matches_closed_form():
    x0, x1, T = 0.0, 2.5, 4.0
    traj = min_snap_1d([x0, x1], [0.0, T])
    for tau in np.linspace(0, T, 25):
        expected = closed_form_min_snap_point_to_point(x0, x1, T, tau)
        got = traj.eval(tau, max_deriv=0)[0]
        assert np.isclose(got, expected, atol=1e-9), (tau, got, expected)


def test_single_segment_boundary_derivatives_are_zero():
    traj = min_snap_1d([0.0, 1.0], [0.0, 3.0])
    start = traj.eval(0.0, max_deriv=3)
    end = traj.eval(3.0, max_deriv=3)
    assert np.allclose(start[1:], 0.0, atol=1e-9)  # vel, acc, jerk = 0 at start
    assert np.allclose(end[1:], 0.0, atol=1e-9)    # vel, acc, jerk = 0 at end
    assert np.isclose(start[0], 0.0, atol=1e-9)
    assert np.isclose(end[0], 1.0, atol=1e-9)


def test_multisegment_hits_waypoints_exactly():
    waypoints = [0.0, 1.0, -0.5, 2.0]
    times = [0.0, 2.0, 5.0, 7.0]
    traj = min_snap_1d(waypoints, times)
    for wp, t in zip(waypoints, times):
        got = traj.eval(t, max_deriv=0)[0]
        assert np.isclose(got, wp, atol=1e-8), (t, got, wp)


def test_multisegment_internal_continuity():
    """Velocity, acceleration, and jerk must match exactly on both sides of
    each internal waypoint (snap itself is allowed to jump -- that's the
    whole point of minimizing its integral, not forcing it continuous)."""
    waypoints = [0.0, 1.0, -0.5, 2.0, 1.5]
    times = [0.0, 1.5, 3.0, 4.0, 6.0]
    traj = min_snap_1d(waypoints, times)
    eps = 1e-6
    for t_internal in times[1:-1]:
        left = traj.eval(t_internal - eps, max_deriv=3)
        right = traj.eval(t_internal + eps, max_deriv=3)
        # loosen tolerance slightly since we're sampling eps off the knot
        assert np.allclose(left[1:], right[1:], atol=1e-3), (t_internal, left, right)


def test_3d_trajectory_with_yaw():
    waypoints_xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.5],
    ])
    times = [0.0, 3.0, 6.0]
    yaw_wp = [0.0, np.pi / 2, np.pi]
    traj = MinimumSnapTrajectory3D(waypoints_xyz, times, yaw_waypoints=yaw_wp)

    for wp, t in zip(waypoints_xyz, times):
        pos = traj.eval_pos(t, max_deriv=0)[0]
        assert np.allclose(pos, wp, atol=1e-8), (t, pos, wp)

    for wp, t in zip(yaw_wp, times):
        yaw = traj.eval_yaw(t, max_deriv=0)[0]
        assert np.isclose(yaw, wp, atol=1e-8), (t, yaw, wp)

    # Start/end at rest (default boundary): velocity/accel/jerk all zero.
    start = traj.eval_pos(0.0, max_deriv=3)
    end = traj.eval_pos(6.0, max_deriv=3)
    assert np.allclose(start[1:], 0.0, atol=1e-8)
    assert np.allclose(end[1:], 0.0, atol=1e-8)


def test_degrees_of_freedom_sanity():
    """order=7 (8 coeffs/segment) must equal 2r-1 for r=4 -- the minimum
    order that supplies enough freedom for position+continuity constraints
    while leaving something for the QP to actually optimize. This isn't a
    behavioral test, it's a guard against silently changing `order` in
    trajectory.py without updating this design invariant."""
    from uav_trajectory_tracking.src.trajectory import min_snap_1d
    traj = min_snap_1d([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert traj.order == 7
