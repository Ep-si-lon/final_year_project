"""Linear stability analysis (Q14): Bode plots + step responses for each
individual control loop, linearized about hover.

Important scope note: the FULL closed-loop trajectory-tracking system
(nonlinear plant + feedforward + feedback, tracking an arbitrary time-
varying trajectory) is not a single LTI system, so "one Bode plot for the
whole thing" isn't a meaningful thing to ask for. What IS meaningful, and
what classical control practice actually does for a cascaded design like
this one, is to linearize and Bode-plot EACH loop independently, about
hover (small-angle, small-rate operating point) -- which is exactly the
operating-point assumption controller.py's gains were designed around in
the first place (see controller.py's docstring on wn/zeta design targets).

Six loops -- see the derivation in each function's docstring:

  1. Z (altitude): thrust deviation -> pz
  2. phi (roll):   L moment -> phi
  3. theta (pitch): M moment -> theta
  4. psi (yaw):    N moment -> psi
  5. x (outer position, via pitch): commanded tilt -> horizontal position
  6. y (outer position, via roll): same, via the roll axis

x and y are now SEPARATE loops (they weren't in the first version of this
file) because the fast-inner-loop fix below makes them genuinely slightly
different: Jxx != Jyy and k_pphi/k_dphi != k_ptheta/k_dtheta (small but
real differences -- see params.py), so the roll-driven y path and the
pitch-driven x path no longer collapse to one identical generic loop once
the real inner-loop dynamics are included.

CORRECTION (found during a post-hoc audit, not in the first version of this
file): loops 1-4's plant included ONLY the rigid-body relation (1/(J*s^2) or
1/(m*s^2)) between COMMANDED moment/thrust and the resulting angle/altitude.
The real signal path also passes through actuator.py's first-order motor
lag, H(s) = 10/(s+10) (tau_motor=0.1s, Snider Eq. 13), which sits between
the controller's commanded [Z,L,M,N] and what the rigid body actually
receives -- verified by directly simulating dynamics.py's rigid_body_deriv
under a small constant moment with the lag bypassed (matches the no-lag
J*angle_ddot=moment prediction to 1.3e-8) versus through the full
plant.py/CrazyflieSim WITH the lag (does NOT match -- confirming the lag is
a real, present pole in the actual system that the first version of this
analysis silently omitted). Quantified impact: at the phi-loop's own
reported gain-crossover frequency (~7.3 rad/s), the omitted lag contributes
an extra ~36 degrees of phase lag -- enough to matter (reported PM dropped
from 80.3deg to ~53deg after the fix below), though not enough to flip any
loop's stability verdict. H(s) is now included in series with the plant for
all 4 PD loops. The outer position loop's P=g/s^2 model does NOT include
this fix -- it never modeled the inner loop's actual dynamics at all (by
design, per the fast-inner-loop assumption above), so there was nothing
inconsistent to add there; that assumption's own accuracy is a separate,
still-open question (see this module's bottom-of-file note).

SECOND CORRECTION (following a user request to stop assuming the inner loop
is instantaneous and actually model it): outer_position_loop() has been
replaced by outer_axis_loop(), which puts the REAL inner attitude closed
loop T_inner(s) (from attitude_loop(), motor lag included) in series with
the g/s^2 kinematic relation, instead of assuming instant/perfect tilt
tracking. This was cross-checked against the actual nonlinear simulation
(a 1m step regulation in x, real controller + real plant): the OLD
fast-inner-loop model predicted 31.1% overshoot / 33.3s settling, actual
was 21.1% / 20.6s -- the old model was not just imprecise, it was
qualitatively the wrong SHAPE of response (assumes an almost-instant initial
rise; the real one is a gentler S-curve, because the real inner loop can't
actually deliver tilt instantly, no matter how "fast" it nominally is).
The new model, by construction, no longer makes that assumption -- it asks
"how much of that S-curve shape does explicitly modeling the real inner
loop's dynamics recover" (see test_stability_analysis.py's cross-check for
the quantitative answer; still not perfect, since the outer model is still
a small-angle linearization of a nonlinear plant, but it removes the one
assumption that was flagged as unverified).

Uses the `control` package (python-control) for margin()/step_info() --
NOT in uav_sim's original scope, added specifically because Q14 asked for
Bode/response plots, which is what that package is for.
"""
import control
import numpy as np

from .controller import ControllerGains
from .params import CrazyflieParams


def _motor_lag(params: CrazyflieParams):
    """H(s) = 10/(s+10) (tau_motor=0.1s), commanded -> actual [Z,L,M,N] --
    see actuator.py. In series with every one of the 4 PD loops' rigid-body
    plant, since the controller's output passes through this before the
    rigid body ever sees it."""
    return control.tf([1.0 / params.tau_motor], [1.0, 1.0 / params.tau_motor])


def z_loop(params: CrazyflieParams, gains: ControllerGains):
    """Plant: m*z_ddot = dZ_actual (thrust deviation from hover) =>
    P(s)=1/(m*s^2), in series with the motor lag (see _motor_lag).
    Controller: PD on altitude error, C(s) = k_dz*s + k_pz."""
    P = control.tf([1.0], [params.mass, 0.0, 0.0]) * _motor_lag(params)
    C = control.tf([gains.k_dz, gains.k_pz], [1.0])
    L = C * P
    T = control.feedback(L, 1)
    return L, T


def attitude_loop(J: float, k_p: float, k_d: float, params: CrazyflieParams):
    """Plant: J*angle_ddot = moment_actual => P(s)=1/(J*s^2), in series with
    the motor lag (see _motor_lag). Controller: PD."""
    P = control.tf([1.0], [J, 0.0, 0.0]) * _motor_lag(params)
    C = control.tf([k_d, k_p], [1.0])
    L = C * P
    T = control.feedback(L, 1)
    return L, T


def outer_axis_loop(params: CrazyflieParams, gains: ControllerGains, inner_loop_result):
    """Plant: x_ddot = g*theta_ACTUAL, and theta_actual is NOT assumed equal
    to the commanded theta_c -- it's whatever the REAL inner attitude loop
    (inner_loop_result, the (L,T) pair from attitude_loop(), motor lag
    included) actually delivers in response to theta_c. So:
        P(s) = g * T_inner(s) / s^2
    replacing the old fast-inner-loop assumption's bare P(s) = g/s^2.
    T_inner already has DC gain 1 (a closed loop with unity feedback tracks
    a constant command exactly at DC), so P(s) still integrates a constant
    commanded tilt into acceleration correctly at low frequency -- the
    difference from the old model only shows up at frequencies where
    T_inner's own dynamics (bandwidth, lag) are no longer close to 1.

    Controller: PID with a FILTERED derivative term -- controller.py's real
    D-term is k_dx*filtered_e_vx, where filtered_e_vx is e_vx passed through
    a first-order low-pass with corner N_filter=10 rad/s (the discrete
    recursion `alpha=N*dt/(1+N*dt)` is the standard bilinear-ish
    discretization of that continuous filter -- confirmed by matching its
    implied corner frequency to N_filter exactly). This is NOT the same as
    a raw derivative k_dx*s -- caught during the fast-inner-loop
    cross-check below (the first version of this function used the raw,
    unfiltered form, which would have contaminated any real-vs-ideal
    comparison with a second, unrelated modeling gap). Since e_vx is
    physically e_x's own exact time-derivative when the target is constant,
    the correct C(s) is the classical "filtered-derivative PID" form:
        C(s) = k_px + k_ix/s + k_dx*(N*s)/(s+N)
    (Z's and the inner attitude loop's D-terms use the RAW error directly,
    e_vz/e_p/e_q/e_r -- no such filter -- confirmed by inspection of
    controller.py's compute(); this correction applies to this loop only.)

    NOTE: with the OLD bare P(s)=g/s^2, this was a type-3 open loop with a
    known-analyzed closed-loop characteristic polynomial
    s^3+2*zeta*wn*s^2+wn^2*s+wn^3, "conditionally stable" (naive gain margin
    reads negative, ~-6dB, even though poles are genuinely stable -- a
    known classical-control artifact for integrator-rich loops). With
    inner_T now in series, the open loop is higher-order and that specific
    closed-form polynomial no longer applies exactly -- but
    test_stability_analysis.py still checks pole locations directly rather
    than trusting the raw GM sign, so this loop is still validated the same
    (robust) way regardless of order."""
    _, T_inner = inner_loop_result
    P = control.tf([params.g], [1.0, 0.0, 0.0]) * T_inner
    N = gains.N_filter
    C = control.tf([gains.k_px, gains.k_ix], [1.0, 0.0]) + control.tf([gains.k_dx * N, 0.0], [1.0, N])
    L = C * P
    T = control.feedback(L, 1)
    return L, T


def outer_x_loop(params: CrazyflieParams, gains: ControllerGains):
    """x is driven via commanded PITCH (theta_c) -- see controller.py's
    `theta_c = ff.theta + (k_px*e_x + ...)`."""
    return outer_axis_loop(params, gains, attitude_loop(params.Jyy, gains.k_ptheta, gains.k_dtheta, params))


def outer_y_loop(params: CrazyflieParams, gains: ControllerGains):
    """y is driven via commanded ROLL (phi_c) -- see controller.py's
    `phi_c = ff.phi - (k_py*e_y + ...)`. The sign flip there is a feedback
    convention (positive roll produces negative y-acceleration, so tracking
    +y needs -phi) and doesn't change this magnitude/phase analysis: a
    negative-feedback loop's linearized transfer function is the same
    whether the physical actuation sign is + or -, since the controller's
    own sign convention already cancels it -- same reasoning the original
    (pre-fix) outer_position_loop() relied on to use a single generic model
    for both axes at all."""
    return outer_axis_loop(params, gains, attitude_loop(params.Jxx, gains.k_pphi, gains.k_dphi, params))


def all_loops(params: CrazyflieParams, gains: ControllerGains) -> dict:
    """Returns {name: (L_open_loop, T_closed_loop)} for all 6 loops."""
    return {
        "Z (altitude)": z_loop(params, gains),
        "phi (roll)": attitude_loop(params.Jxx, gains.k_pphi, gains.k_dphi, params),
        "theta (pitch)": attitude_loop(params.Jyy, gains.k_ptheta, gains.k_dtheta, params),
        "psi (yaw)": attitude_loop(params.Jzz, gains.k_ppsi, gains.k_dpsi, params),
        "x (outer position, via pitch)": outer_x_loop(params, gains),
        "y (outer position, via roll)": outer_y_loop(params, gains),
    }


def margins_and_step_info(L, T) -> dict:
    gm, pm, wg, wp = control.margin(L)
    gm_db = 20 * np.log10(gm) if np.isfinite(gm) else np.inf
    info = control.step_info(T)
    return {
        "gain_margin_db": gm_db,
        "phase_margin_deg": pm,
        "gain_crossover_freq_rad_s": wp,
        "phase_crossover_freq_rad_s": wg,
        "rise_time_s": info["RiseTime"],
        "settling_time_s": info["SettlingTime"],
        "overshoot_pct": info["Overshoot"],
        "peak": info["Peak"],
    }
