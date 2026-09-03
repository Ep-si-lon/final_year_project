"""Computes gain/phase margins for all 6 loops (Section 6, Eq. 14-19) using
this project's own controller gains and dynamics parameters, and compares
against the report's own stated Section 6 results (p.12-16) -- a third,
frequency-domain validation of the plant+controller, alongside the
time-domain checks in test_closed_loop_reproduction.py.

Not a pytest test (an infinite gain margin doesn't assert cleanly, and the
report itself calls the outer-loop margins "volatile and sensitive to small
changes" -- so this is presented as a comparison table, with
test_stability_margins.py holding the tolerance-bounded regression
assertions that DO make sense to assert). Run directly:
`python3 uav_sim/validation_tests/compare_stability_margins.py`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uav_sim.src.controller import gains_from_report_design_parameters
from uav_sim.src.params import CrazyflieParams
from uav_sim.src.stability_analysis import ALL_LOOPS, margins_db_deg

# Report's own Section 6 results (p.12-16), (gain_margin_dB or None, phase_margin_deg, freq_rad_s for PM)
REPORT_RESULTS = {
    "Z": (None, 46.0, 5.46),
    "L_inner": (None, 46.7, 6.23),
    "L_outer": (7.33, 53.3, 6.82),
    "M_inner": (None, 46.0, 7.33),
    "M_outer": (6.76, 56.3, 7.64),
    "N": (None, 50.8, 5.38),
}


def main():
    params = CrazyflieParams()
    gains = gains_from_report_design_parameters(params)

    print(f"{'Loop':<10}{'Ours GM[dB]':<14}{'Report GM[dB]':<16}"
          f"{'Ours PM[deg]':<14}{'Report PM[deg]':<16}{'Ours wPM':<10}{'Report wPM'}")
    for name, (fn, eq_ref) in ALL_LOOPS.items():
        L = fn(params, gains)
        gm_db, pm, wcg, wcp = margins_db_deg(L)
        rep_gm, rep_pm, rep_w = REPORT_RESULTS[name]
        gm_str = f"{gm_db:.2f}" if gm_db is not None else "inf"
        rep_gm_str = f"{rep_gm:.2f}" if rep_gm is not None else "inf"
        print(f"{name:<10}{gm_str:<14}{rep_gm_str:<16}"
              f"{pm:<14.2f}{rep_pm:<16.2f}{wcp:<10.2f}{rep_w:.2f}")


if __name__ == "__main__":
    main()
