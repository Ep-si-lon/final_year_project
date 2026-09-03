"""Regression test locking in the qualitative match between this project's
closed-loop plant+controller and Snider Figures 12/13 (p.20-21), reusing
reproduce_snider_figures.py's setup exactly. Not an exact numeric match to
the PDF's own plot (no raw data available, only a rendered image to eyeball)
-- these assertions capture the qualitative behavior the report itself
describes: pz rises and settles at 0.5m by ~2.5s with no overshoot (Fig. 12),
and px/py show a small, nonzero, monotonic drift rather than either
perfect station-keeping or runaway divergence (Fig. 13, "unsuccessful
control of the px and py drift ... unable to completely mitigate the
steady-state error").
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


def _check_drift_matches_figure_13(log):
    px, py = log[:, 0], log[:, 1]
    # small but nonzero -- not perfect station-keeping, not runaway divergence
    assert 1e-4 < abs(px[-1]) < 0.2
    assert 1e-4 < abs(py[-1]) < 0.2
    # report's own plot shows roughly monotonic drift, not oscillation
    assert np.all(np.diff(px) >= -1e-6) or np.all(np.diff(px) <= 1e-6)
    assert np.all(np.diff(py) >= -1e-6) or np.all(np.diff(py) <= 1e-6)


def test_literal_reproduction_matches_figures_12_and_13():
    params = CrazyflieParams()
    gammas = compute_gammas(params)
    log = run_literal(params, gammas)
    _check_pz_matches_figure_12(log)
    _check_drift_matches_figure_13(log)


def test_full_plant_reproduction_matches_figures_12_and_13():
    params = CrazyflieParams()
    log = run_full_plant(params)
    _check_pz_matches_figure_12(log)
    _check_drift_matches_figure_13(log)


def test_literal_and_full_plant_agree_with_each_other():
    """The motor lag (present in full_plant, absent in literal -- see
    reproduce_snider_figures.py module docstring) should perturb the result
    only modestly for this near-hover, slow-drift scenario."""
    params = CrazyflieParams()
    gammas = compute_gammas(params)
    log_literal = run_literal(params, gammas)
    log_full = run_full_plant(params)
    assert abs(log_literal[-1, 2] - log_full[-1, 2]) < 0.01  # pz
    assert abs(log_literal[-1, 0] - log_full[-1, 0]) < 0.02  # px
    assert abs(log_literal[-1, 1] - log_full[-1, 1]) < 0.02  # py
