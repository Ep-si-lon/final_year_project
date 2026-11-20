"""Cascaded PD/PID feedback + differential-flatness feedforward tracking
controller (Q4/D3: NOT the full SE(3) geometric controller -- Euler-angle
cascade, matching this project's "gentle/indoor" envelope decision).

Structure, per waypoint the same as uav_sim/src/controller.py's
CascadedController, but every reference is now TIME-VARYING (from a
MinimumSnapTrajectory3D + flatness.compute_feedforward) instead of a fixed
ref6, and every loop's command is feedforward + feedback rather than
feedback-only:

    Z     = Z_ff(t)                    + k_pz*e_z     + k_dz*e_vz
    theta_c = theta_ff(t)              + k_px*e_x + k_ix*int(e_x) + k_dx*e_vx_filt
    phi_c   = phi_ff(t)                - (k_py*e_y + k_iy*int(e_y) + k_dy*e_vy_filt)
    L = k_pphi*(theta_c - phi)   + k_dphi*(p_ff(t) - p)     [see NOTE below on phi/theta naming]
    M = k_ptheta*(theta_c - theta) + k_dtheta*(q_ff(t) - q)
    N = k_ppsi*(psi_ff(t) - psi) + k_dpsi*(r_ff(t) - r)

Gains (ControllerGains, gains_from_report_design_parameters,
gains_pole_placement_outer) are forked verbatim from uav_sim/src/controller.py
-- same physical vehicle, same "gentle/indoor" envelope, so the same
zeta/wn design targets apply; see that module's docstring for the full
derivation and the report-vs-corrected-dynamics rationale.

Output saturation (Q12, default: yes) is new relative to uav_sim's
controller: [Z_min,Z_max] and per-axis [-tau_max,tau_max] limits are applied
to the final command, using the same bounds uav_sim's own
validation_tests/test_fuzz.py already treats as "physically reasonable"
(Z in [0, 2*m*g]; L,M in +-5e-4 N*m; N in +-3e-4 N*m) -- NOT real hardware
motor limits. Per ASSUMPTIONS_AND_QUESTIONS.md's hardware-deployment item
(Q2), replacing these with actual Crazyflie motor-derived limits (and adding
a real PWM mixer) is flagged as a deferred follow-up, not done here.
"""
import dataclasses
from dataclasses import dataclass

import numpy as np

from .dynamics import rotation_body_to_earth
from .flatness import compute_feedforward
from .params import CrazyflieParams
from .trajectory import MinimumSnapTrajectory3D


@dataclass(frozen=True)
class ControllerGains:
    k_pz: float = 0.297
    k_dz: float = 0.198

    k_pphi: float = 1.4914539e-4
    k_dphi: float = 1.19316312e-4
    k_ptheta: float = 1.499004180e-4
    k_dtheta: float = 1.499004180e-4
    k_ppsi: float = 1.82885325e-4
    k_dpsi: float = 1.75569912e-4

    k_px: float = 0.2
    k_ix: float = 0.1
    k_dx: float = 0.1
    k_py: float = 0.2
    k_iy: float = 0.1
    k_dy: float = 0.1

    N_filter: float = 10.0


@dataclass(frozen=True)
class SaturationLimits:
    Z_min: float = 0.0
    Z_max: float = None  # filled in from params (2*m*g) if left None
    L_max: float = 5e-4
    M_max: float = 5e-4
    N_max: float = 3e-4


def gains_from_report_design_parameters(params: CrazyflieParams) -> ControllerGains:
    zeta_z, wn_z = 1.0, 3.0
    zeta_roll, wn_roll = 1.2, 3.0
    zeta_pitch, wn_pitch = 1.5, 3.0
    zeta_yaw, wn_yaw = 1.2, 2.5
    return ControllerGains(
        k_pz=params.mass * wn_z ** 2,
        k_dz=params.mass * 2 * zeta_z * wn_z,
        k_pphi=params.Jxx * wn_roll ** 2,
        k_dphi=params.Jxx * 2 * zeta_roll * wn_roll,
        k_ptheta=params.Jyy * wn_pitch ** 2,
        k_dtheta=params.Jyy * 2 * zeta_pitch * wn_pitch,
        k_ppsi=params.Jzz * wn_yaw ** 2,
        k_dpsi=params.Jzz * 2 * zeta_yaw * wn_yaw,
    )


def gains_pole_placement_outer(
    params: CrazyflieParams,
    base: ControllerGains = None,
    wn_outer: float = 0.6,
    zeta_outer: float = 1.0,
) -> ControllerGains:
    if base is None:
        base = gains_from_report_design_parameters(params)
    kp = wn_outer ** 2 / params.g
    ki = wn_outer ** 3 / params.g
    kd = 2 * zeta_outer * wn_outer / params.g
    return dataclasses.replace(base, k_px=kp, k_ix=ki, k_dx=kd, k_py=kp, k_iy=ki, k_dy=kd)


class TrajectoryTrackingController:
    """Stateful: instantiate once per trajectory run, call .compute(state12,
    traj, t) once per control step (integral_x/y and the filtered velocity
    error persist across calls -- see uav_sim/src/controller.py's docstring
    on why a fresh instance per step would silently discard the I-term)."""

    def __init__(
        self,
        params: CrazyflieParams = None,
        gains: ControllerGains = None,
        dt: float = 0.002,
        integral_clamp: float = 0.5,
        saturation: SaturationLimits = None,
    ):
        self.params = params if params is not None else CrazyflieParams()
        self.gains = gains if gains is not None else ControllerGains()
        self.dt = dt
        self.integral_clamp = integral_clamp
        if saturation is None:
            saturation = SaturationLimits()
        if saturation.Z_max is None:
            saturation = dataclasses.replace(saturation, Z_max=2.0 * self.params.mass * self.params.g)
        self.saturation = saturation

        self.integral_x = 0.0
        self.integral_y = 0.0
        self.filtered_e_vx = 0.0
        self.filtered_e_vy = 0.0
        self._alpha = self.gains.N_filter * dt / (1.0 + self.gains.N_filter * dt)

    def reset(self) -> None:
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.filtered_e_vx = 0.0
        self.filtered_e_vy = 0.0

    def compute(self, state12: np.ndarray, traj: MinimumSnapTrajectory3D, t: float) -> np.ndarray:
        """Returns commanded [Z, L, M, N] at trajectory time t given the
        current 12-state."""
        g = self.gains
        px, py, pz = state12[0], state12[1], state12[2]
        u, v, w = state12[3], state12[4], state12[5]
        phi, theta, psi = state12[6], state12[7], state12[8]
        p, q, r = state12[9], state12[10], state12[11]

        R = rotation_body_to_earth(phi, theta, psi)
        vx, vy, vz = R @ np.array([u, v, w])

        ff = compute_feedforward(traj, t, self.params)
        pos_vel_des = traj.eval_pos(t, max_deriv=1)
        pos_des, vel_des = pos_vel_des[0], pos_vel_des[1]

        e_x = pos_des[0] - px
        e_y = pos_des[1] - py
        e_z = pos_des[2] - pz
        e_vx = vel_des[0] - vx
        e_vy = vel_des[1] - vy
        e_vz = vel_des[2] - vz

        self.integral_x += e_x * self.dt
        self.integral_y += e_y * self.dt
        if self.integral_clamp is not None:
            self.integral_x = np.clip(self.integral_x, -self.integral_clamp, self.integral_clamp)
            self.integral_y = np.clip(self.integral_y, -self.integral_clamp, self.integral_clamp)

        self.filtered_e_vx = (1 - self._alpha) * self.filtered_e_vx + self._alpha * e_vx
        self.filtered_e_vy = (1 - self._alpha) * self.filtered_e_vy + self._alpha * e_vy

        theta_c = ff.theta + (g.k_px * e_x + g.k_ix * self.integral_x + g.k_dx * self.filtered_e_vx)
        phi_c = ff.phi - (g.k_py * e_y + g.k_iy * self.integral_y + g.k_dy * self.filtered_e_vy)

        Z = ff.Z + g.k_pz * e_z + g.k_dz * e_vz

        e_phi = phi_c - phi
        e_theta = theta_c - theta
        e_psi = ff.psi - psi
        e_p = ff.p - p
        e_q = ff.q - q
        e_r = ff.r - r

        L = g.k_pphi * e_phi + g.k_dphi * e_p
        M = g.k_ptheta * e_theta + g.k_dtheta * e_q
        N = g.k_ppsi * e_psi + g.k_dpsi * e_r

        s = self.saturation
        Z = np.clip(Z, s.Z_min, s.Z_max)
        L = np.clip(L, -s.L_max, s.L_max)
        M = np.clip(M, -s.M_max, s.M_max)
        N = np.clip(N, -s.N_max, s.N_max)

        return np.array([Z, L, M, N])
