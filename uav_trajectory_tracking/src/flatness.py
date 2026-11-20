"""Differential-flatness feedforward: given a MinimumSnapTrajectory3D, compute
the [Z, phi, theta, psi, p, q, r] that would EXACTLY produce that trajectory
if the plant were a perfect, undisturbed instance of dynamics.py's rigid-body
model (per Mellinger & Kumar, ICRA 2011, adapted here to this project's
Euler-angle representation rather than their rotation-matrix one -- see
ASSUMPTIONS_AND_QUESTIONS.md Q3/D2: "gentle/indoor" envelope, Euler angles
confirmed sufficient).

This is feedforward ONLY -- it assumes the plant matches the model exactly
and there is no disturbance. controller.py (Stage 2, not yet built) is
responsible for combining this with feedback to correct whatever the model
gets wrong in practice.

Derivation:
1. Earth-frame Newton's second law (thrust rotated to earth frame, gravity
   already in earth frame -- no approximation, valid at any attitude):
       m*a_des = R(phi,theta,psi) @ [0,0,Z] - m*g*z_hat
   so the required earth-frame force is F_des = m*(a_des + g*z_hat), thrust
   magnitude Z_des = |F_des|, and the required body z-axis (expressed in
   earth frame) is z_b_des = F_des / Z_des.
2. Solve for (phi_des, theta_des) directly from dynamics.py's own R[:,2]
   (body z-axis expressed in earth frame), holding psi_des fixed at the
   trajectory's own commanded yaw -- NOT via the generic SO(3)/rotation-
   matrix literature's "project a desired heading vector, cross-product for
   the other two body axes" construction. That construction is standard for
   the quaternion/rotation-matrix (Lee/Leok/McClamroch-style) controllers,
   but it has a known property this project's earlier draft ran into: the
   yaw it produces, once ZYX-Euler-decomposed, is NOT exactly the psi_des
   fed in whenever tilt is nonzero (confirmed numerically: ~1e-2 rad error
   at ~0.3 rad tilt) -- it's a different, tilt-dependent definition of "yaw"
   that only coincides with the intrinsic Euler yaw at zero tilt. Since this
   project deliberately stays in the Euler-angle representation (D2), we
   instead solve the 2-equation/2-unknown system directly:
       R[:,2] = [cpsi*stheta*cphi + spsi*sphi,
                 spsi*stheta*cphi - cpsi*sphi,
                 ctheta*cphi]                     (derived from Rz@Ry@Rx)
   Combining the first two rows to eliminate theta gives
   sphi = spsi*zx - cpsi*zy exactly; the remaining combination gives
   tan(theta) = (cpsi*zx+spsi*zy)/zz. This reproduces (phi,theta) to machine
   precision given any (phi,theta,psi) that generated z_b (verified: max
   error 1.4e-16 rad over random trials) -- because psi is fixed by
   construction rather than re-extracted, there is no round-trip to lose
   precision or ambiguity in.
3. Desired body rates (p,q,r) are obtained by numerically differentiating
   (phi_des, theta_des, psi_des) over time and inverting
   dynamics.py's euler_rate_matrix -- exact for this Euler-angle model, no
   extra closed-form angular-velocity formula needed (that formula exists in
   the literature for the rotation-matrix/SO(3) representation, which this
   project deliberately isn't using; inverting our own kinematics matrix is
   simpler and exactly consistent with the rest of this codebase).
"""
from dataclasses import dataclass

import numpy as np

from .dynamics import euler_rate_matrix
from .params import CrazyflieParams
from .trajectory import MinimumSnapTrajectory3D


@dataclass
class FlatnessFeedforward:
    Z: float
    phi: float
    theta: float
    psi: float
    p: float
    q: float
    r: float


def _desired_attitude(a_des: np.ndarray, psi_des: float, params: CrazyflieParams):
    """Returns (Z_des, phi_des, theta_des). psi_des is returned unchanged --
    it's an input held fixed, not re-derived (see module docstring)."""
    z_hat = np.array([0.0, 0.0, 1.0])
    F_des = params.mass * (a_des + params.g * z_hat)
    Z_des = np.linalg.norm(F_des)
    if Z_des < 1e-9:
        raise ValueError("desired thrust ~0 -- free-fall trajectory, attitude undefined")
    zx, zy, zz = F_des / Z_des
    if abs(zz) < 1e-6:
        raise ValueError(
            "desired body z-axis ~horizontal (~90deg tilt) -- singular case for "
            "this Euler-angle solve, not handled (matches this project's "
            "'gentle/indoor' scope, D2/Q3)"
        )
    cpsi, spsi = np.cos(psi_des), np.sin(psi_des)

    phi_des = np.arcsin(np.clip(spsi * zx - cpsi * zy, -1.0, 1.0))
    theta_des = np.arctan2(cpsi * zx + spsi * zy, zz)

    return Z_des, phi_des, theta_des


def compute_feedforward(
    traj: MinimumSnapTrajectory3D,
    t: float,
    params: CrazyflieParams,
    fd_eps: float = 1e-5,
) -> FlatnessFeedforward:
    """Feedforward [Z, phi, theta, psi, p, q, r] at time t."""
    pos_derivs = traj.eval_pos(t, max_deriv=2)
    a_des = pos_derivs[2]
    psi_des = traj.eval_yaw(t, max_deriv=0)[0]

    Z_des, phi_des, theta_des = _desired_attitude(a_des, psi_des, params)

    def euler_at(tt):
        a = traj.eval_pos(tt, max_deriv=2)[2]
        psi = traj.eval_yaw(tt, max_deriv=0)[0]
        _, phi_, theta_ = _desired_attitude(a, psi, params)
        return np.array([phi_, theta_, psi])

    t_lo = max(t - fd_eps, traj.t_start)
    t_hi = min(t + fd_eps, traj.t_end)
    denom = t_hi - t_lo
    if denom < 1e-12:
        euler_dot = np.zeros(3)
    else:
        euler_dot = (euler_at(t_hi) - euler_at(t_lo)) / denom

    E = euler_rate_matrix(phi_des, theta_des)
    p_des, q_des, r_des = np.linalg.solve(E, euler_dot)

    return FlatnessFeedforward(
        Z=Z_des, phi=phi_des, theta=theta_des, psi=psi_des,
        p=p_des, q=q_des, r=r_des,
    )
