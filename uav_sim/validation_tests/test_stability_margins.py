"""Regression test for the frequency-domain validation in
compare_stability_margins.py: all 6 loops' gain/phase margins and phase-margin
crossover frequency match Snider Section 6's stated results (p.12-16) closely
(within rounding of the report's own 3-sig-fig printed values), once the
outer loops include the extra `/s` term present in Listing 3's actual MATLAB
code (`OLy = PID_tf*CLL/s`, p.18) but missing from the printed Eq. 16/18 --
see CHANGELOG.md.
"""
import numpy as np
import pytest

from uav_sim.src.controller import gains_from_report_design_parameters
from uav_sim.src.params import CrazyflieParams
from uav_sim.src.stability_analysis import ALL_LOOPS, margins_db_deg

# (gain_margin_dB or None for infinite, phase_margin_deg, freq_rad_s at PM)
REPORT_RESULTS = {
    "Z": (None, 46.0, 5.46),
    "L_inner": (None, 46.7, 6.23),
    "L_outer": (7.33, 53.3, 6.82),
    "M_inner": (None, 46.0, 7.33),
    "M_outer": (6.76, 56.3, 7.64),
    "N": (None, 50.8, 5.38),
}


@pytest.mark.parametrize("name", list(ALL_LOOPS.keys()))
def test_loop_matches_report_stability_margins(name):
    params = CrazyflieParams()
    gains = gains_from_report_design_parameters(params)
    fn, _ = ALL_LOOPS[name]
    gm_db, pm, wcg, wcp = margins_db_deg(fn(params, gains))
    rep_gm, rep_pm, rep_w = REPORT_RESULTS[name]

    if rep_gm is None:
        assert gm_db is None or gm_db > 40  # "infinite" in practice
    else:
        assert abs(gm_db - rep_gm) < 0.5

    assert abs(pm - rep_pm) < 1.0
    assert abs(wcp - rep_w) < 0.1
