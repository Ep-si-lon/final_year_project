"""Validation trajectory set (Q7): straight line, circle, helix. Each is
built the SAME way the general minimum-snap generator is meant to be used in
practice -- sample waypoints along the ideal geometric path, then fit a
minimum-snap polynomial through them (src/trajectory.py) -- rather than
hard-coding a separate closed-form parametric trajectory type. This means
these functions exercise the actual generator machinery, not a shortcut
around it.

FIX (found by inspection after the fact -- the first version of this file
produced a visibly non-circular "circle"): fitting minimum-snap through only
~12 waypoints/lap with a REST-TO-REST (zero velocity/accel/jerk) boundary
condition at both ends does not reproduce a circle. Two separate problems,
both fixed below:

1. Too few waypoints: with only 12 points around the loop, the polynomial
   between waypoints is free to minimize snap however it likes subject only
   to position + continuity -- nothing tells it to bow along the true arc in
   between. Measured: radius varied 0.93-1.09 on a nominal r=1.0 circle
   (~15% distortion), and the top-down XY shape was a visibly wobbly
   rounded-dodecagon, not a circle. Fix: sample far more densely (36+
   waypoints/lap) so the fit is forced to hug the true arc everywhere, not
   just at 12 vertices.
2. Rest-to-rest boundary is physically wrong for a continuous loop: real
   uniform circular motion has constant nonzero tangential velocity
   EVERYWHERE, including at the point chosen as "start/end" -- forcing zero
   velocity there produces a visible outward spike/cusp right at the seam
   (a real stop-and-restart transient, not a rendering artifact). Fix: for
   an integer number of laps, the analytic velocity/acceleration/jerk of
   uniform circular motion is IDENTICAL at t=0 and t=T (same angle mod 2pi)
   -- so feeding that analytic value in as the boundary condition (instead
   of the zero default) makes the fitted trajectory loop back on itself
   smoothly, with no artificial start/stop transient, exactly matching what
   "fly a circle" should mean physically.
"""
import numpy as np

from .trajectory import MinimumSnapTrajectory3D


def straight_line_trajectory(p0, p1, T: float, yaw0: float = 0.0, yaw1: float = 0.0) -> MinimumSnapTrajectory3D:
    """Point-to-point, start/end at rest (default boundary conditions) --
    rest-to-rest IS physically correct here, unlike for the closed loops
    below: a straight-line move genuinely starts and ends stationary."""
    waypoints = np.array([p0, p1], dtype=float)
    times = [0.0, T]
    return MinimumSnapTrajectory3D(waypoints, times, yaw_waypoints=[yaw0, yaw1])


def _circular_motion_boundary(radius: float, omega: float, theta0: float) -> np.ndarray:
    """Analytic (velocity, acceleration, jerk) of uniform circular motion
    (angle = omega*t + theta0) at the given angle -- used as the boundary
    condition for x and y so the fit starts/ends already moving along the
    true tangent instead of from rest. Returns shape (3, 2) (deriv x, y)."""
    c, s = np.cos(theta0), np.sin(theta0)
    vel = radius * omega * np.array([-s, c])
    acc = radius * omega ** 2 * np.array([-c, -s])
    jerk = radius * omega ** 3 * np.array([s, -c])
    return np.array([vel, acc, jerk])


def circle_trajectory(
    center=(0.0, 0.0, 0.5),
    radius: float = 1.0,
    n_laps: float = 1.0,
    T: float = 20.0,
    n_waypoints_per_lap: int = 36,
    yaw_fixed: float = 0.0,
) -> MinimumSnapTrajectory3D:
    """Circle in the horizontal plane at constant height, constant angular
    rate. n_laps MUST be an integer for the smooth-loop boundary condition
    below to be exact (start/end angle then coincide mod 2pi)."""
    if not np.isclose(n_laps, round(n_laps)):
        raise ValueError(f"n_laps must be an integer for a closed, smoothly-looping circle, got {n_laps}")
    center = np.asarray(center, dtype=float)
    n_wp = max(int(round(n_waypoints_per_lap * n_laps)) + 1, 3)
    omega = 2 * np.pi * n_laps / T
    angles = np.linspace(0.0, 2 * np.pi * n_laps, n_wp)
    times = np.linspace(0.0, T, n_wp)
    waypoints = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
        np.full(n_wp, center[2]),
    ])
    yaw_wp = np.full(n_wp, yaw_fixed)

    xy_boundary = _circular_motion_boundary(radius, omega, theta0=0.0)  # same at t=0 and t=T (integer laps)
    boundary_start = np.column_stack([xy_boundary, np.zeros(3)])  # z: at rest (constant height)
    boundary_end = boundary_start.copy()

    return MinimumSnapTrajectory3D(
        waypoints, times, yaw_waypoints=yaw_wp,
        boundary_xyz_start=boundary_start, boundary_xyz_end=boundary_end,
    )


def helix_trajectory(
    center_xy=(0.0, 0.0),
    radius: float = 1.0,
    z0: float = 0.2,
    z1: float = 1.5,
    n_laps: float = 2.0,
    T: float = 25.0,
    n_waypoints_per_lap: int = 36,
    yaw_fixed: float = 0.0,
) -> MinimumSnapTrajectory3D:
    """Circle in x/y (constant angular rate) combined with a linear climb in
    z (constant climb rate) -- both start and end already in steady motion
    (circular tangential velocity in x/y, constant climb rate in z), not at
    rest, matching how a real helical flight path behaves throughout,
    including at its two ends. n_laps must be an integer -- see
    circle_trajectory's docstring."""
    if not np.isclose(n_laps, round(n_laps)):
        raise ValueError(f"n_laps must be an integer, got {n_laps}")
    cx, cy = center_xy
    n_wp = max(int(round(n_waypoints_per_lap * n_laps)) + 1, 3)
    omega = 2 * np.pi * n_laps / T
    angles = np.linspace(0.0, 2 * np.pi * n_laps, n_wp)
    times = np.linspace(0.0, T, n_wp)
    z = np.linspace(z0, z1, n_wp)
    waypoints = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
        z,
    ])
    yaw_wp = np.full(n_wp, yaw_fixed)

    xy_boundary = _circular_motion_boundary(radius, omega, theta0=0.0)
    climb_rate = (z1 - z0) / T
    z_boundary = np.array([climb_rate, 0.0, 0.0])  # constant velocity, zero accel/jerk
    boundary_start = np.column_stack([xy_boundary, z_boundary])
    boundary_end = boundary_start.copy()

    return MinimumSnapTrajectory3D(
        waypoints, times, yaw_waypoints=yaw_wp,
        boundary_xyz_start=boundary_start, boundary_xyz_end=boundary_end,
    )
