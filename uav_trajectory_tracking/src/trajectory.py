"""Minimum-derivative (minimum-snap for x/y/z, minimum-acceleration for yaw)
piecewise-polynomial trajectory generator, per Mellinger & Kumar, "Minimum
Snap Trajectory Generation and Control for Quadrotors" (ICRA 2011), and the
closed-form QP formulation described in Bry's "Minimum Snap Trajectory
Generation" tutorial notes.

Per ASSUMPTIONS_AND_QUESTIONS.md (D4/Q6): general minimum-snap through
arbitrary waypoints+times, not hand-coded parametric curves.

Math, for one axis independently (x, y, z, and yaw all solved separately --
the minimum-snap cost and the position/continuity constraints are
axis-separable, so there is no loss of generality in solving 4 independent
1D problems and recombining):

For N waypoints there are m = N-1 segments. Segment i is a polynomial of
degree `order` in LOCAL time tau in [0, dt_i] (dt_i = duration of segment i,
NOT normalized to [0,1] -- see test_trajectory.py for the numerical-
conditioning check that justifies this simpler choice for this project's
typical segment durations):

    p_i(tau) = sum_{k=0}^{order} c_{i,k} * tau^k

We minimize integral_0^{dt_i} (d^r p_i / dtau^r)^2 dtau summed over all
segments (r=4 for snap on x/y/z, r=2 for acceleration on yaw), subject to:
  - p_i(0) / p_i(dt_i) match the given waypoint positions
  - derivatives 1..r-1 (velocity..jerk for r=4) are pinned to given boundary
    values at the very first and very last waypoint (default: zero, i.e.
    start/end at rest)
  - derivatives 1..r-1 are CONTINUOUS across each internal waypoint (their
    actual values there are free -- solved for by the QP, not fixed)

This is a quadratic program with linear equality constraints, solved via the
standard KKT linear system (no iterative optimizer needed):

    [Q  A^T] [c]   [0]
    [A  0  ] [l] = [b]

`order = 2*r - 1` gives exactly enough per-segment coefficients (order+1) to
satisfy all the above constraints while leaving the remaining freedom for
the QP to optimize -- see test_trajectory.py's degree-of-freedom check.
"""
import math
from dataclasses import dataclass

import numpy as np


def _deriv_row(order: int, m: int, tau: float) -> np.ndarray:
    """Row vector `row` such that row @ c = d^m p / dtau^m at local time tau,
    for p(tau) = sum c_k tau^k, k=0..order."""
    row = np.zeros(order + 1)
    for k in range(m, order + 1):
        coeff = math.factorial(k) // math.factorial(k - m)
        row[k] = coeff * tau ** (k - m)
    return row


def _cost_matrix(order: int, r: int, dt: float) -> np.ndarray:
    """Q such that c^T Q c = integral_0^dt (d^r p/dtau^r)^2 dtau."""
    Q = np.zeros((order + 1, order + 1))
    for i in range(r, order + 1):
        ci = math.factorial(i) // math.factorial(i - r)
        for j in range(r, order + 1):
            cj = math.factorial(j) // math.factorial(j - r)
            power = i + j - 2 * r + 1
            Q[i, j] = ci * cj * dt ** power / power
    return Q


def _solve_min_derivative_1d(
    waypoints: np.ndarray,
    times: np.ndarray,
    r: int,
    order: int,
    boundary_start: np.ndarray = None,
    boundary_end: np.ndarray = None,
) -> np.ndarray:
    """Returns coeffs, shape (m, order+1), one row per segment.

    boundary_start/boundary_end: length r-1 arrays giving derivatives 1..r-1
    (velocity, acceleration, ... ) at the first/last waypoint. Default: all
    zero (start and end at rest).
    """
    waypoints = np.asarray(waypoints, dtype=float)
    times = np.asarray(times, dtype=float)
    n_wp = len(waypoints)
    m = n_wp - 1
    if m < 1:
        raise ValueError("need at least 2 waypoints (1 segment)")
    if boundary_start is None:
        boundary_start = np.zeros(r - 1)
    if boundary_end is None:
        boundary_end = np.zeros(r - 1)

    durations = np.diff(times)
    ncoef = order + 1
    nvar = ncoef * m

    Q = np.zeros((nvar, nvar))
    for i in range(m):
        sl = slice(i * ncoef, (i + 1) * ncoef)
        Q[sl, sl] = _cost_matrix(order, r, durations[i])

    rows = []
    rhs = []

    def add_row(seg_idx: int, deriv: int, tau: float, value: float, seg_idx2=None, tau2=None, sign2=-1.0):
        row = np.zeros(nvar)
        row[seg_idx * ncoef:(seg_idx + 1) * ncoef] = _deriv_row(order, deriv, tau)
        if seg_idx2 is not None:
            row[seg_idx2 * ncoef:(seg_idx2 + 1) * ncoef] += sign2 * _deriv_row(order, deriv, tau2)
        rows.append(row)
        rhs.append(value)

    # Waypoint position constraints: start of segment 0, end of every segment,
    # AND start of every internal segment (i>=1) -- without this last one,
    # nothing pins segment i's start to the shared waypoint value; only its
    # end was constrained, leaving its start free to drift (caught by
    # test_multisegment_hits_waypoints_exactly).
    add_row(0, 0, 0.0, waypoints[0])
    for i in range(m):
        add_row(i, 0, durations[i], waypoints[i + 1])
    for i in range(1, m):
        add_row(i, 0, 0.0, waypoints[i])

    # Boundary derivative constraints at the very first and last waypoint.
    for d in range(1, r):
        add_row(0, d, 0.0, boundary_start[d - 1])
        add_row(m - 1, d, durations[m - 1], boundary_end[d - 1])

    # Internal continuity: derivatives 1..r-1 match across each internal waypoint.
    for i in range(1, m):
        for d in range(1, r):
            # (end of segment i-1) - (start of segment i) = 0
            add_row(i - 1, d, durations[i - 1], 0.0, seg_idx2=i, tau2=0.0, sign2=-1.0)

    A = np.array(rows)
    b = np.array(rhs)

    n_con = A.shape[0]
    KKT = np.zeros((nvar + n_con, nvar + n_con))
    KKT[:nvar, :nvar] = Q
    KKT[:nvar, nvar:] = A.T
    KKT[nvar:, :nvar] = A
    rhs_full = np.concatenate([np.zeros(nvar), b])

    sol = np.linalg.solve(KKT, rhs_full)
    c = sol[:nvar]
    return c.reshape(m, ncoef)


@dataclass
class Trajectory1D:
    """One flat output (x, y, z, or yaw) as a function of time."""
    coeffs: np.ndarray   # (m, order+1)
    times: np.ndarray    # (m+1,) cumulative waypoint times, strictly increasing
    order: int

    def eval(self, t: float, max_deriv: int) -> np.ndarray:
        """Returns array of length max_deriv+1: [value, d/dt, d2/dt2, ...]."""
        t = min(max(t, self.times[0]), self.times[-1])
        m = self.coeffs.shape[0]
        seg = int(np.searchsorted(self.times, t, side="right") - 1)
        seg = min(max(seg, 0), m - 1)
        tau = t - self.times[seg]
        out = np.zeros(max_deriv + 1)
        for d in range(max_deriv + 1):
            out[d] = _deriv_row(self.order, d, tau) @ self.coeffs[seg]
        return out


def min_snap_1d(waypoints, times, boundary_start=None, boundary_end=None) -> Trajectory1D:
    """r=4 (snap), order=7 -- for x, y, z."""
    coeffs = _solve_min_derivative_1d(waypoints, times, r=4, order=7,
                                       boundary_start=boundary_start, boundary_end=boundary_end)
    return Trajectory1D(coeffs=coeffs, times=np.asarray(times, dtype=float), order=7)


def min_accel_1d(waypoints, times, boundary_start=None, boundary_end=None) -> Trajectory1D:
    """r=2 (acceleration), order=5 -- for yaw (only needs continuity through
    velocity for the flatness feedforward; order 5 leaves margin so yaw is at
    least jerk-continuous too)."""
    coeffs = _solve_min_derivative_1d(waypoints, times, r=2, order=5,
                                       boundary_start=boundary_start, boundary_end=boundary_end)
    return Trajectory1D(coeffs=coeffs, times=np.asarray(times, dtype=float), order=5)


class MinimumSnapTrajectory3D:
    """Minimum-snap x/y/z + minimum-acceleration yaw, evaluated together."""

    def __init__(
        self,
        waypoints_xyz: np.ndarray,
        times: np.ndarray,
        yaw_waypoints: np.ndarray = None,
        boundary_xyz_start: np.ndarray = None,
        boundary_xyz_end: np.ndarray = None,
    ):
        waypoints_xyz = np.asarray(waypoints_xyz, dtype=float)
        times = np.asarray(times, dtype=float)
        if yaw_waypoints is None:
            yaw_waypoints = np.zeros(len(times))

        def bnd(b, axis):
            return None if b is None else np.asarray(b, dtype=float)[:, axis]

        self.traj = [
            min_snap_1d(waypoints_xyz[:, ax], times,
                        boundary_start=bnd(boundary_xyz_start, ax),
                        boundary_end=bnd(boundary_xyz_end, ax))
            for ax in range(3)
        ]
        self.traj_yaw = min_accel_1d(yaw_waypoints, times)
        self.t_start = times[0]
        self.t_end = times[-1]

    def eval_pos(self, t: float, max_deriv: int = 3) -> np.ndarray:
        """Returns (max_deriv+1, 3): rows are position, velocity, accel, jerk, ..."""
        return np.column_stack([self.traj[ax].eval(t, max_deriv) for ax in range(3)])

    def eval_yaw(self, t: float, max_deriv: int = 1) -> np.ndarray:
        """Returns (max_deriv+1,): [psi, psi_dot, ...]."""
        return self.traj_yaw.eval(t, max_deriv)
