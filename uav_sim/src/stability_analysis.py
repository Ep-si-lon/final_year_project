"""Open-loop transfer functions and stability margins, per Snider Section 6
(p.12-16), Eq. 14-19. Uses the `control` package (python-control).

Each loop's open-loop TF at the plant input includes the motor transfer
function H(s)=10/(s+10) (Eq. 13). The two outer loops (roll/pitch position)
close their respective inner attitude loop first (unity feedback,
`Lo_inner/(1+Lo_inner)`) before cascading with the outer PID, exactly
matching Eq. 16/18's `Lo_inner*(I+Lo_inner)^-1` term.
"""
import control
import numpy as np

from .controller import ControllerGains, gains_from_report_design_parameters
from .params import CrazyflieParams

MOTOR_TF = control.tf([10], [1, 10])  # H(s) = 10/(s+10), Eq. 13

s = control.tf("s")


def z_open_loop(params: CrazyflieParams, gains: ControllerGains):
    """Eq. 14: Loz = (kdz*s + kpz)/(m*s^2) * 10/(s+10)."""
    return (gains.k_dz * s + gains.k_pz) / (params.mass * s ** 2) * MOTOR_TF


def phi_inner_open_loop(params: CrazyflieParams, gains: ControllerGains):
    """Eq. 15: Lo_phi,inner = (kdphi*s + kpphi)/(Jxx*s^2) * 10/(s+10)."""
    return (gains.k_dphi * s + gains.k_pphi) / (params.Jxx * s ** 2) * MOTOR_TF


def phi_outer_open_loop(params: CrazyflieParams, gains: ControllerGains):
    """Eq. 16 as printed reads Lo_phi,outer = g*(kdy*s^2+kpy*s+kiy)/(s*(1+s/N))
    * Lo_inner/(1+Lo_inner) -- but Listing 3's own MATLAB (p.18, `OLy =
    PID_tf*CLL/s`) has an EXTRA `/s` beyond the printed equation. Without it,
    the computed margins/crossover frequency don't match Fig. 8's stated
    values at all (crossover ~26 rad/s vs stated 6.82 rad/s); with it, they
    match closely. Implemented per the code (which is what actually produced
    Fig. 8), not the printed equation. See CHANGELOG.md."""
    inner = phi_inner_open_loop(params, gains)
    closed_inner = control.feedback(inner, 1)
    outer_pid = params.g * (gains.k_dy * s ** 2 + gains.k_py * s + gains.k_iy) / (
        s * (1 + s / gains.N_filter)
    )
    return outer_pid * closed_inner / s


def theta_inner_open_loop(params: CrazyflieParams, gains: ControllerGains):
    """Eq. 17: Lo_theta,inner = (kdtheta*s + kptheta)/(Jyy*s^2) * 10/(s+10)."""
    return (gains.k_dtheta * s + gains.k_ptheta) / (params.Jyy * s ** 2) * MOTOR_TF


def theta_outer_open_loop(params: CrazyflieParams, gains: ControllerGains):
    """Eq. 18: same extra `/s` issue as phi_outer_open_loop -- see its
    docstring and CHANGELOG.md."""
    inner = theta_inner_open_loop(params, gains)
    closed_inner = control.feedback(inner, 1)
    outer_pid = params.g * (gains.k_dx * s ** 2 + gains.k_px * s + gains.k_ix) / (
        s * (1 + s / gains.N_filter)
    )
    return outer_pid * closed_inner / s


def psi_open_loop(params: CrazyflieParams, gains: ControllerGains):
    """Eq. 19: Lopsi = (kdpsi*s + kppsi)/(Jzz*s^2) * 10/(s+10)."""
    return (gains.k_dpsi * s + gains.k_ppsi) / (params.Jzz * s ** 2) * MOTOR_TF


def margins_db_deg(open_loop_tf):
    """(gain margin [dB, None if infinite], phase margin [deg],
    gain crossover freq [rad/s], phase crossover freq [rad/s])."""
    gm, pm, wcg, wcp = control.margin(open_loop_tf)
    gm_db = 20 * np.log10(gm) if np.isfinite(gm) and gm > 0 else None
    return gm_db, pm, wcg, wcp


ALL_LOOPS = {
    "Z": (z_open_loop, "Eq. 14, p.12"),
    "L_inner": (phi_inner_open_loop, "Eq. 15, p.13"),
    "L_outer": (phi_outer_open_loop, "Eq. 16, p.13"),
    "M_inner": (theta_inner_open_loop, "Eq. 17, p.14"),
    "M_outer": (theta_outer_open_loop, "Eq. 18, p.15"),
    "N": (psi_open_loop, "Eq. 19, p.16"),
}
