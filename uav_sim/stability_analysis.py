"""Frequency-domain stability margins for each control loop, mirroring
Section 6 of the WashU ESE4481 report (Bode/Nyquist gain & phase margins),
built from THIS project's actual gains and motor-lag model rather than
copying their numbers.

Unlike the report, our outer-loop transfer functions have no derivative
low-pass filter term (their "1/(1+s/N)") -- we feed back the true velocity
state directly instead of differentiating a noisy discrete position error,
so there's no derivative-kick/noise problem to filter away.

Target (same as the reference report): gain margin > 6 dB, phase margin > 45 deg.

Two presets are checked (GAIN_PRESETS in controller.py): "aggressive" (best
tracking, fails several margin checks) and "margin_safe" (passes Z/phi/theta/
psi cleanly). Both presets still show a weak outer x/y-position gain margin
(~-5 dB) -- this matches the reference report's own experience: their outer
position-loop gains were, in their words, "cumbersome" to derive from the
wn/zeta formula and were hand-tuned instead, and their discussion section
recommends LQR as a better fit for that loop specifically. We report this as
a known, expected limitation of the PID/PD architecture rather than chase it
further -- an LQR outer loop is a natural next improvement, not a bug fix.
"""

import control
import numpy as np
from dynamics import CrazyflieParams
from controller import CascadedPIDController, GAIN_PRESETS
from actuator import MotorLag

GM_TARGET_DB = 6.0
PM_TARGET_DEG = 45.0


def build_loops(gain_preset="aggressive"):
    params = CrazyflieParams()
    ctrl = CascadedPIDController(params, **GAIN_PRESETS[gain_preset])
    motor_bw = MotorLag().omega  # 10 rad/s, matches the reference report's H(s)

    s = control.tf("s")
    H_motor = motor_bw / (s + motor_bw)

    Jxx, Jyy, Jzz = params.Ixx, params.Iyy, params.Izz

    # Z: controller.py's g_z gains are ACCELERATION-domain (scale=1 -- the
    # /mass conversion happens explicitly in F=mass*(G+az_cmd)/cos(..)), and
    # the plant divides by mass again (z_ddot = F/mass - g) -- those two
    # mass factors cancel exactly, so no mass term belongs in this loop at all.
    L_oz = (ctrl.g_z.kd * s + ctrl.g_z.kp) / s ** 2 * H_motor

    # Attitude: controller.py's g_phi/g_theta/g_psi gains are TORQUE-domain
    # (scale=Ixx/Iyy/Izz -- tau_phi is output directly, no further scaling in
    # compute()), so the plant's own physics (phi_ddot = tau/Ixx) is a
    # separate, real division that must appear here.
    L_ophi_inner = (ctrl.g_phi.kd * s + ctrl.g_phi.kp) / (Jxx * s ** 2) * H_motor
    L_otheta_inner = (ctrl.g_theta.kd * s + ctrl.g_theta.kp) / (Jyy * s ** 2) * H_motor
    L_opsi = (ctrl.g_psi.kd * s + ctrl.g_psi.kp) / (Jzz * s ** 2) * H_motor

    inner_phi_closed = control.feedback(L_ophi_inner, 1)
    inner_theta_closed = control.feedback(L_otheta_inner, 1)

    C_x = (ctrl.g_x.kd * s ** 2 + ctrl.g_x.kp * s + ctrl.g_x.ki) / s
    C_y = (ctrl.g_y.kd * s ** 2 + ctrl.g_y.kp * s + ctrl.g_y.ki) / s

    # roll (phi) drives y; pitch (theta) drives x -- matches controller.py's
    # small-angle inversion (phi <-> y, theta <-> x). Note: no extra factor
    # of g here -- controller.py divides by g to turn ax_cmd into theta_des,
    # and the physical plant multiplies back by g (x_ddot = g*theta), so the
    # g's cancel and the outer loop is just inner_closed(s)/s^2.
    L_oy_outer = C_y * inner_phi_closed / s ** 2
    L_ox_outer = C_x * inner_theta_closed / s ** 2

    return {
        "Z (thrust)": L_oz,
        "phi inner (roll rate/angle)": L_ophi_inner,
        "theta inner (pitch rate/angle)": L_otheta_inner,
        "psi (yaw)": L_opsi,
        "y outer (via roll)": L_oy_outer,
        "x outer (via pitch)": L_ox_outer,
    }


def report(gain_preset):
    print(f"\n=== gain preset: {gain_preset} ===")
    loops = build_loops(gain_preset)
    print(f"{'Loop':32s} {'GM (dB)':>10s} {'PM (deg)':>10s}  Verdict")
    print("-" * 70)
    for name, L in loops.items():
        gm, pm, wg, wp = control.margin(L)
        gm_db = 20 * np.log10(gm) if np.isfinite(gm) and gm > 0 else np.inf
        ok = (gm_db > GM_TARGET_DB or not np.isfinite(gm_db)) and pm > PM_TARGET_DEG
        verdict = "OK" if ok else "MARGIN TOO LOW"
        gm_str = "inf" if not np.isfinite(gm_db) else f"{gm_db:.2f}"
        print(f"{name:32s} {gm_str:>10s} {pm:>10.2f}  {verdict}")


if __name__ == "__main__":
    for preset in GAIN_PRESETS:
        report(preset)

    print(f"\nTargets: gain margin > {GM_TARGET_DB} dB, phase margin > {PM_TARGET_DEG} deg "
          "(same thresholds as the reference report).")
