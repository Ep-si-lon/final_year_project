"""Regression test for reproduce_snider_figures.py's setup, post dynamics-fix
(see that module's docstring and CHANGELOG.md for why this no longer
reproduces Fig. 13's px/py drift): pz should still rise and settle at 0.5m
by ~2.5s with no overshoot (Fig. 12 -- unaffected by the fix, since it's a
pure-hover, no-tilt scenario). px/py, given the small initial tilt
disturbance, should show a bounded, damped correction back toward
station-keeping near the origin -- NOT Fig. 13's drift (an artifact of the
report's own missing tilt-to-translate coupling, now fixed) and NOT runaway
divergence (which is what happens if `ControllerGains`'s default,
report-literal outer gains are used post-fix instead of
`gains_pole_placement_outer`).
"""
import numpy as np

from uav_sim.validation_tests.reproduce_snider_figures import (
    DT, STEPS, run_full_plant, run_literal,
)
from uav_sim.src.params import CrazyflieParams, compute_gammas


def _check_pz_matches_figure_12(log):
    t = np.linspace(0, STEPS * DT, STEPS + 1)
    pz = log[:, 2]
    assert abs(pz[-1] - 0.5) < 0.01
    settled = pz[t >= 2.5]
    assert np.all((settled > 0.49) & (settled < 0.51))
    assert pz.max() < 0.55  # no significant overshoot


def _check_station_keeping(log):
    px, py = log[:, 0], log[:, 1]
    # bounded response to the small initial tilt disturbance, not divergence
    assert np.abs(px).max() < 0.2
    assert np.abs(py).max() < 0.2
    # damped correction, ending up close to the origin (not Fig. 13's drift)
    assert abs(px[-1]) < 0.05
    assert abs(py[-1]) < 0.05


def test_literal_reproduction():
    params = CrazyflieParams()
    gammas = compute_gammas(params)
    log = run_literal(params, gammas)
    _check_pz_matches_figure_12(log)
    _check_station_keeping(log)


def test_full_plant_reproduction():
    params = CrazyflieParams()
    log = run_full_plant(params)
    _check_pz_matches_figure_12(log)
    _check_station_keeping(log)


def test_literal_and_full_plant_agree_with_each_other():
    """The motor lag (present in full_plant, absent in literal -- see
    reproduce_snider_figures.py module docstring) should perturb the result
    only modestly for this near-hover, bounded-oscillation scenario."""
    params = CrazyflieParams()
    gammas = compute_gammas(params)
    log_literal = run_literal(params, gammas)
    log_full = run_full_plant(params)
    assert abs(log_literal[-1, 2] - log_full[-1, 2]) < 0.01  # pz
    assert abs(log_literal[-1, 0] - log_full[-1, 0]) < 0.03  # px
    assert abs(log_literal[-1, 1] - log_full[-1, 1]) < 0.03  # py
