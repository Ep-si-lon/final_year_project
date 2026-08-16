"""The 'Controller' block: takes error e = r - y, produces control input u.

Cascaded PID/PD architecture, gains derived from pole-placement formulas
(natural frequency omega_n + damping ratio zeta per loop) following the
WashU ESE4481 CrazyFlie 2.1 report, Section 5.2:

    Z, inner attitude (phi, theta), yaw (psi):  PD, direct rate feedback
        kp = J * wn^2          kd = J * 2*zeta*wn      (J = mass for Z)
    outer position (x, y):  PID, direct velocity feedback, in the
    acceleration domain (the /g conversion to a commanded tilt angle is a
    separate explicit step below, matching this module's existing structure)
        kp = wn^2   ki = wn^3   kd = 2*zeta*wn

Rate/velocity feedback (using the true state, not a finite difference of
the error) avoids "derivative kick" when the setpoint itself is moving --
the same reason the reference report reads gyro rate directly instead of
differentiating the attitude error.
"""

import numpy as np
from dynamics import G, CrazyflieParams


class Integrator:
    """Plain accumulator with anti-windup clamping, for the outer PID's I-term."""

    def __init__(self, limit=None):
        self.value = 0.0
        self.limit = limit

    def reset(self):
        self.value = 0.0

    def step(self, error, dt):
        self.value += error * dt
        if self.limit is not None:
            self.value = np.clip(self.value, -self.limit, self.limit)
        return self.value


class LoopGains:
    """Pole-placement gain derivation, Section 5.2 of the reference report."""

    def __init__(self, wn, zeta, scale=1.0, ki=False):
        self.kp = scale * wn ** 2
        self.kd = scale * 2 * zeta * wn
        self.ki = scale * wn ** 3 if ki else 0.0


# Two gain presets, deliberately kept side by side rather than picking one --
# see stability_analysis.py for the frequency-domain numbers behind each:
#
#   "aggressive"   circle-tracking error 0.83 cm, attitude-loop phase margin
#                  13.8 deg (target >45 deg, FAILS). Great nominal performance,
#                  but only ~14 ms of unmodeled delay/model error before the
#                  attitude loop can start oscillating -- fine for a pure
#                  simulation study, risky if ever pointed at real hardware.
#   "margin_safe"  circle-tracking error ~20 cm, attitude-loop phase margin
#                  46.7 deg (PASSES), matching the reference report's own
#                  target and its roll-inner-loop result almost exactly.
#                  ~131 ms of tolerable unmodeled delay -- an order of
#                  magnitude more robustness margin, at a real cost in
#                  tracking accuracy.
#   "hand_tuned"   same Z/attitude/yaw as margin_safe (already pass), but the
#                  outer x/y loop gains are hand-picked (outer_kp/ki/kd below)
#                  instead of coming from the wn_xy/zeta_xy formula -- exactly
#                  what the reference report did for this exact loop, after
#                  finding the formula "cumbersome" there. A grid search over
#                  kp/ki/kd found the integral term is what was destroying the
#                  outer loop's margin (every passing combination had ki~0);
#                  kp=0.3, ki=0.02, kd=0.8 clears both targets by a wide
#                  margin (GM=18.7 dB, PM=64.4 deg) while keeping a little
#                  integral action for drift rejection.
#   "reference_replica"  a literal copy of the report's own Listing 2/3 numbers
#                  -- not our formula-derived gains, THEIR five hand-picked
#                  values per cascaded axis (wn_inner, zeta_inner, kp_outer,
#                  ki_outer, kd_outer), including their asymmetric roll vs.
#                  pitch inner loop (zeta_roll=1.2 vs zeta_pitch=1.5, a
#                  discrepancy in the report itself despite its text claiming
#                  "the same bandwidth and damping" for both) and their
#                  outer-loop derivative low-pass filter (N=10), which their
#                  report says was "necessary due to instability from
#                  amplifying errors" when differentiating a noisy position
#                  error. Use this preset as the fidelity baseline; build
#                  improvements (like margin_safe/hand_tuned) on top of it.
GAIN_PRESETS = {
    "aggressive": dict(wn_z=3.0, zeta_z=1.0, wn_xy=2.2, zeta_xy=0.9,
                        wn_attitude=13.0, zeta_attitude=1.3, wn_yaw=6.0, zeta_yaw=1.2),
    "margin_safe": dict(wn_z=3.0, zeta_z=1.0, wn_xy=0.5, zeta_xy=0.9,
                         wn_attitude=3.0, zeta_attitude=1.2, wn_yaw=3.0, zeta_yaw=1.2),
    "hand_tuned": dict(wn_z=3.0, zeta_z=1.0, wn_xy=0.5, zeta_xy=0.9,
                        wn_attitude=3.0, zeta_attitude=1.2, wn_yaw=3.0, zeta_yaw=1.2,
                        outer_gains=dict(kp=0.3, ki=0.02, kd=0.8)),
    "reference_replica": dict(
        wn_z=3.0, zeta_z=1.0,
        wn_phi=3.0, zeta_phi=1.2,       # L (roll) inner loop, Sec 6.2
        wn_theta=3.0, zeta_theta=1.5,   # M (pitch) inner loop, Sec 6.3
        wn_yaw=2.5, zeta_yaw=1.2,       # N (yaw), Sec 6.4
        outer_gains=dict(kp=0.2, ki=0.1, kd=0.1),  # kpx=kpy, kix=kiy, kdx=kdy, Listing 2
        deriv_filter_N=10.0,            # Listing 2's "N_filter = 10" derivative low-pass
    ),
    # PSO-optimized (pso_tune.py), Z/yaw held at margin_safe's values in all
    # three -- only the loops actually in question (inner roll/pitch, outer
    # x/y) were searched. All three independently converged to ki_outer~=0,
    # confirming hand_tuned's grid-search finding that integral action is
    # what destroys this loop's margin. Several best values sit exactly on a
    # search bound (kd_outer=2.0 here, kp_outer=1.0 in pso_5var_raw,
    # zeta_inner=2.0 in pso_5var_formula) -- these are the best PSO found
    # within the bounds searched, not necessarily a true unconstrained optimum.
    "pso_3var": dict(wn_z=3.0, zeta_z=1.0, wn_yaw=3.0, zeta_yaw=1.2,
                      wn_attitude=3.0, zeta_attitude=1.2,
                      outer_gains=dict(kp=0.9124758772940267, ki=0.0, kd=2.0)),
    "pso_5var_formula": dict(wn_z=3.0, zeta_z=1.0, wn_yaw=3.0, zeta_yaw=1.2,
                              wn_attitude=2.772591296019356, zeta_attitude=2.0,
                              outer_gains=dict(kp=0.404428459350719, ki=0.009966483625753524,
                                                kd=1.542711618942076)),
    "pso_5var_raw": dict(wn_z=3.0, zeta_z=1.0, wn_yaw=3.0, zeta_yaw=1.2,
                          inner_gains=dict(kp=0.00012334409476751009, kd=7.406661793004076e-05),
                          outer_gains=dict(kp=1.0, ki=0.0, kd=1.7316026182531068)),
}


class CascadedPIDController:
    """Outer position loop (PID) -> desired attitude/thrust,
    inner attitude loop (PD, rate feedback) -> body torques.

    Bandwidths below follow the reference report's guidance: inner loop
    (attitude) bandwidth should be ~5-10x the outer loop (position)
    bandwidth, or the outer loop sees phase lag from a "slow" inner loop.
    Defaults to the "aggressive" preset -- pass gains=GAIN_PRESETS["margin_safe"]
    (unpacked) for the frequency-margin-justified alternative.
    """

    def __init__(self, params=None,
                 wn_z=3.0, zeta_z=1.0,
                 wn_xy=2.2, zeta_xy=0.9,
                 wn_attitude=13.0, zeta_attitude=1.3,
                 wn_phi=None, zeta_phi=None, wn_theta=None, zeta_theta=None,
                 wn_yaw=6.0, zeta_yaw=1.2,
                 i_limit=2.0, outer_gains=None, deriv_filter_N=None, inner_gains=None):
        self.p = params if params is not None else CrazyflieParams()

        # wn_phi/zeta_phi (roll) and wn_theta/zeta_theta (pitch) override
        # wn_attitude/zeta_attitude independently, for presets (like the
        # reference report) that tune the two inner loops asymmetrically.
        wn_phi = wn_attitude if wn_phi is None else wn_phi
        zeta_phi = zeta_attitude if zeta_phi is None else zeta_phi
        wn_theta = wn_attitude if wn_theta is None else wn_theta
        zeta_theta = zeta_attitude if zeta_theta is None else zeta_theta

        # Derivative low-pass filter (report Listing 2's "N_filter"), applied
        # to the outer x/y velocity-feedback term. Off by default -- this
        # project reads true velocity state rather than differentiating a
        # noisy position error, so there's normally no derivative kick to
        # filter away; the reference_replica preset turns it on to match
        # the report's Sec 6.2/6.3 outer-loop transfer functions exactly.
        self.deriv_filter_N = deriv_filter_N
        self.filt_evx = 0.0
        self.filt_evy = 0.0

        # Z: PD only (no integral) -- thrust's gravity feedforward already
        # removes the steady-state bias an integral term would otherwise fix.
        self.g_z = LoopGains(wn_z, zeta_z)

        # Outer position loops: PID, acceleration domain (no /g here --
        # the /g happens explicitly in the small-angle inversion below).
        # outer_gains, if given, bypasses the wn_xy/zeta_xy formula entirely
        # with directly hand-picked {kp, ki, kd} -- mirrors the reference
        # report abandoning the formula for this specific loop.
        if outer_gains is not None:
            self.g_x = LoopGains.__new__(LoopGains)
            self.g_x.kp, self.g_x.ki, self.g_x.kd = outer_gains["kp"], outer_gains["ki"], outer_gains["kd"]
            self.g_y = LoopGains.__new__(LoopGains)
            self.g_y.kp, self.g_y.ki, self.g_y.kd = outer_gains["kp"], outer_gains["ki"], outer_gains["kd"]
        else:
            self.g_x = LoopGains(wn_xy, zeta_xy, ki=True)
            self.g_y = LoopGains(wn_xy, zeta_xy, ki=True)
        self.int_x = Integrator(limit=i_limit)
        self.int_y = Integrator(limit=i_limit)

        # Inner attitude loops: PD, torque domain, scaled by inertia.
        # inner_gains, if given, bypasses the wn_phi/zeta_phi (and
        # wn_theta/zeta_theta) formula entirely with directly hand-picked
        # {kp, kd} applied symmetrically to both roll and pitch -- used by
        # pso_tune.py's raw-gain search variant.
        if inner_gains is not None:
            self.g_phi = LoopGains.__new__(LoopGains)
            self.g_phi.kp, self.g_phi.kd, self.g_phi.ki = inner_gains["kp"], inner_gains["kd"], 0.0
            self.g_theta = LoopGains.__new__(LoopGains)
            self.g_theta.kp, self.g_theta.kd, self.g_theta.ki = inner_gains["kp"], inner_gains["kd"], 0.0
        else:
            self.g_phi = LoopGains(wn_phi, zeta_phi, scale=self.p.Ixx)
            self.g_theta = LoopGains(wn_theta, zeta_theta, scale=self.p.Iyy)

        # Yaw: PD only, torque domain.
        self.g_psi = LoopGains(wn_yaw, zeta_yaw, scale=self.p.Izz)

        self.max_tilt = np.deg2rad(25)  # limit commanded roll/pitch

    def compute(self, state, ref_pos, dt, ref_yaw=0.0, ref_vel=None, ref_acc=None):
        """state: 12-vector from QuadrotorDynamics.
        ref_pos: desired [x, y, z] -- this is 'r' in the diagram.
        ref_vel, ref_acc: desired [vx,vy,vz] / [ax,ay,az] from the reference
            trajectory (feedforward + velocity-feedback target). Default to
            zero, which reduces to pure feedback around a fixed point.
        returns (u, e_pos) where u = [F, tau_phi, tau_theta, tau_psi]."""
        if ref_vel is None:
            ref_vel = np.zeros(3)
        if ref_acc is None:
            ref_acc = np.zeros(3)

        pos = state[0:3]
        vel = state[3:6]
        phi, theta, psi = state[6:9]
        p_rate, q_rate, r_rate = state[9:12]

        e_pos = ref_pos - pos      # this is 'e' in the diagram
        e_vel = ref_vel - vel

        # ---- Z: PD + feedforward, direct vertical-velocity feedback ----
        az_cmd = ref_acc[2] + self.g_z.kp * e_pos[2] + self.g_z.kd * e_vel[2]
        F = self.p.mass * (G + az_cmd) / (np.cos(phi) * np.cos(theta) + 1e-6)

        # ---- Outer X/Y: PID + feedforward, direct velocity feedback ----
        ix = self.int_x.step(e_pos[0], dt)
        iy = self.int_y.step(e_pos[1], dt)

        e_vel_x, e_vel_y = e_vel[0], e_vel[1]
        if self.deriv_filter_N is not None:
            alpha = self.deriv_filter_N * dt / (1 + self.deriv_filter_N * dt)
            self.filt_evx = (1 - alpha) * self.filt_evx + alpha * e_vel_x
            self.filt_evy = (1 - alpha) * self.filt_evy + alpha * e_vel_y
            e_vel_x, e_vel_y = self.filt_evx, self.filt_evy

        ax_cmd = ref_acc[0] + self.g_x.kp * e_pos[0] + self.g_x.ki * ix + self.g_x.kd * e_vel_x
        ay_cmd = ref_acc[1] + self.g_y.kp * e_pos[1] + self.g_y.ki * iy + self.g_y.kd * e_vel_y

        # small-angle inversion: desired horizontal accel -> desired tilt
        phi_des = np.clip(
            (ax_cmd * np.sin(ref_yaw) - ay_cmd * np.cos(ref_yaw)) / G,
            -self.max_tilt, self.max_tilt)
        theta_des = np.clip(
            (ax_cmd * np.cos(ref_yaw) + ay_cmd * np.sin(ref_yaw)) / G,
            -self.max_tilt, self.max_tilt)

        # ---- Inner attitude: PD, direct gyro-rate feedback (target rate 0) ----
        e_phi = phi_des - phi
        e_theta = theta_des - theta
        tau_phi = self.g_phi.kp * e_phi - self.g_phi.kd * p_rate
        tau_theta = self.g_theta.kp * e_theta - self.g_theta.kd * q_rate

        # ---- Yaw: PD, direct gyro-rate feedback (target rate 0) ----
        e_psi = ref_yaw - psi
        tau_psi = self.g_psi.kp * e_psi - self.g_psi.kd * r_rate

        u = np.array([F, tau_phi, tau_theta, tau_psi])
        return u, e_pos
