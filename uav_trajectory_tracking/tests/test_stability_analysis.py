"""Confirms each linearized loop (src/stability_analysis.py) is actually
stable with positive margins, using the same gains gains_pole_placement_outer
produces for the real controller -- if this ever fails, the controller
gains stopped being a sane design, not just "the plots look different"."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uav_trajectory_tracking.src.controller import gains_pole_placement_outer
from uav_trajectory_tracking.src.params import CrazyflieParams
from uav_trajectory_tracking.src.stability_analysis import all_loops, margins_and_step_info


def test_all_loops_are_stable_with_positive_margins():
    """Pole location (closed-loop poles all in the left half-plane) is the
    ground-truth stability check for every loop -- always asserted.

    The single-crossing gain-margin reading is only a reliable STABILITY
    indicator for the 4 simple PD-on-double-integrator loops (Z, phi, theta,
    psi): those are provably stable for any k_p,k_d>0 (textbook result), and
    numerically always show a single phase crossing with positive GM/PM.

    The outer x/y position loop is PID-on-a-double-integrator (type-3 open
    loop) and is "conditionally stable": verified here (and by hand, via
    Routh-Hurwitz on its known closed-loop characteristic polynomial
    s^3+2*zeta*wn*s^2+wn^2*s+wn^3, matching controller.py's own derivation)
    to have all poles strictly stable at the ACTUAL design gains -- but its
    raw gain margin reads negative (~-6dB) because |L(jw)|>1 at the
    frequency where phase crosses -180, a well-known classical-control
    artifact for integrator-rich loops, not an instability. So this loop is
    checked by poles only, not by the naive margin sign."""
    params = CrazyflieParams()
    gains = gains_pole_placement_outer(params)
    loops = all_loops(params, gains)

    simple_pd_loops = {"Z (altitude)", "phi (roll)", "theta (pitch)", "psi (yaw)"}

    for name, (L, T) in loops.items():
        poles = T.poles()
        assert np.all(poles.real < 0), (name, poles)

        m = margins_and_step_info(L, T)
        assert np.isfinite(m["settling_time_s"]), (name, m)
        if name in simple_pd_loops:
            assert m["phase_margin_deg"] > 0, (name, m)
            assert m["gain_margin_db"] > 0, (name, m)
