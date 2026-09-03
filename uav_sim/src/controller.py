"""Cascaded PID/PD controller, per Snider Section 5 and Listing 2 (p.10-11).

Z (thrust) and N (yaw) are single PD loops. L (roll) and M (pitch) are
cascaded: an outer PID on lateral position/velocity produces a commanded tilt
angle (small-angle inversion, p.9 Eq. trim relations), fed to an inner PD on
attitude/rate.

Gains are computed from Listing 2's own zeta/w_n design parameters, not
copied from Table 1 (p.29) -- Table 1's k_pphi "Simulation" entry
(1.4194e-4) is an apparent digit-transposition typo: every other tabulated
gain matches Jxx/Jyy/Jzz*w_n^2 (or *2*zeta*w_n) to 4-5 significant figures,
but k_pphi = Jxx*w_n_roll^2 with Listing 2's own w_n_roll=3.0 computes to
1.4914e-4, not 1.4194e-4. See CHANGELOG.md.

Reference `ref6 = [px_c, py_c, pz_c, vx_c, vy_c, vz_c]` is in the EARTH
frame (Eq. 2 setpoint), matching Listing 4's own `ref` vector. Since this
plant's state stores u,v,w in the BODY frame (Section 2), velocities are
rotated to earth frame here before comparing against ref6 -- Listing 4's own
script instead keeps a world-frame velocity state and only rotates into body
frame right before calling the dynamics function (and back out again after),
an internal inconsistency in that script; rotating once, here, at the
controller boundary is the physically correct place to do it and gives the
same near-hover behavior Listing 4's script does.

Listing 2's own code computes `e_p = 0 - p` then `L = k_pphi*e_phi +
k_dphi*e_p` (i.e. `-k_dphi*p`), which is the OPPOSITE sign from Section
5.1's prose ("L = ... + k_dphi*p_hat"). The code is what actually produced
Fig. 12/13, so it's what's implemented here; see CHANGELOG.md.
"""
from dataclasses import dataclass

import numpy as np

from .dynamics import rotation_body_to_earth
from .params import CrazyflieParams


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


def gains_from_report_design_parameters(params: CrazyflieParams) -> ControllerGains:
    """Recompute the gains directly from Listing 2's zeta/w_n design
    parameters and this plant's own Jxx/Jyy/Jzz/mass, rather than the
    hardcoded ControllerGains() defaults (useful if params ever change)."""
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


class CascadedController:
    """Stateful controller: instantiate once, call .compute(state12, ref6)
    once per control step (matches Listing 2's per-call signature, which
    threads integral_x/integral_y/filtered state through explicitly)."""

    def __init__(
        self,
        params: CrazyflieParams = None,
        gains: ControllerGains = None,
        dt: float = 0.002,
        integral_clamp: float = None,
    ):
        """integral_clamp: None (default) reproduces Listing 2 exactly --
        no anti-windup, matching the report's own MATLAB "Simulation"
        controller used for Fig. 12/13/Section 6. Over long runs (~100s+)
        this lets integral_x/integral_y grow without bound whenever a
        steady-state position error persists, which in turn drives phi/theta
        to diverge (verified: reaches ~2 rad by t=100s for the report's own
        hover+disturbance scenario) -- a real property of the report's own
        controller, not a bug in this implementation. Passing a value (e.g.
        0.5, matching the firmware's own `constrain(...,-0.5,0.5)` in
        Listing 6, p.28) clamps integral_x/integral_y to +-that value each
        step, for long-duration simulations where bounded behavior is
        wanted instead of a literal Listing-2 reproduction."""
        self.params = params if params is not None else CrazyflieParams()
        self.gains = gains if gains is not None else ControllerGains()
        self.dt = dt
        self.integral_clamp = integral_clamp
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

    def compute(self, state12: np.ndarray, ref6: np.ndarray) -> np.ndarray:
        """Returns commanded [Z, L, M, N]. ref6 = [px_c,py_c,pz_c,vx_c,vy_c,vz_c]."""
        g = self.gains
        px, py, pz = state12[0], state12[1], state12[2]
        u, v, w = state12[3], state12[4], state12[5]
        phi, theta, psi = state12[6], state12[7], state12[8]
        p, q, r = state12[9], state12[10], state12[11]

        R = rotation_body_to_earth(phi, theta, psi)
        vx, vy, vz = R @ np.array([u, v, w])

        e_x = ref6[0] - px
        e_y = ref6[1] - py
        e_z = ref6[2] - pz
        e_vx = ref6[3] - vx
        e_vy = ref6[4] - vy
        e_vz = ref6[5] - vz

        self.integral_x += e_x * self.dt
        self.integral_y += e_y * self.dt
        if self.integral_clamp is not None:
            self.integral_x = np.clip(self.integral_x, -self.integral_clamp, self.integral_clamp)
            self.integral_y = np.clip(self.integral_y, -self.integral_clamp, self.integral_clamp)

        self.filtered_e_vx = (1 - self._alpha) * self.filtered_e_vx + self._alpha * e_vx
        self.filtered_e_vy = (1 - self._alpha) * self.filtered_e_vy + self._alpha * e_vy

        theta_c = g.k_px * e_x + g.k_ix * self.integral_x + g.k_dx * self.filtered_e_vx
        phi_c = -(g.k_py * e_y + g.k_iy * self.integral_y + g.k_dy * self.filtered_e_vy)

        Z = self.params.mass * self.params.g + g.k_pz * e_z + g.k_dz * e_vz

        e_phi = phi_c - phi
        e_theta = theta_c - theta
        e_psi = 0.0 - psi
        e_p = 0.0 - p
        e_q = 0.0 - q
        e_r = 0.0 - r

        L = g.k_pphi * e_phi + g.k_dphi * e_p
        M = g.k_ptheta * e_theta + g.k_dtheta * e_q
        N = g.k_ppsi * e_psi + g.k_dpsi * e_r

        return np.array([Z, L, M, N])
