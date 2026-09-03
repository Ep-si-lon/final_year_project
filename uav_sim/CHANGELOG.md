# Changelog

## 2026-09-03 — Fixed missing tilt-to-translate physics; redesigned outer-loop gains

Attempting a circle trajectory (radius=1m, z=10m) surfaced a fundamental
gap, not a tuning problem: the report's own printed translational EOM
(p.2) is `(u_dot,v_dot,w_dot) = (rv-qw,pw-ru,qu-pv) + (0,0,(Z-mg)/m)` --
gravity subtracted only from the body-z component, with no rotation
applied. That's the near-hover approximation (valid only at phi=theta=0),
presented as if it were the general nonlinear equation. Verified
experimentally: commanding and holding a 14 deg pitch tilt at rest produced
**exactly zero** horizontal acceleration -- tilt-to-translate, the entire
mechanism the controller's own design depends on (`p_double_dot_y=g*phi`,
stated explicitly in Section 5.1's text), doesn't exist in the report's own
equations. Cross-checked: the report's own printed linearization (Eq. 5,
p.3) has an all-zero row for u_dot too -- so Section 3's linearization,
Section 5's control design, and Section 6's stability analysis are all
self-consistent with each other, but built on dynamics that can't actually
translate via tilt. This also recasts the earlier "successful reproduction"
of Fig. 13's px/py drift: that drift was real (bit-for-bit same equations)
but driven by a secondary numerical artifact (residual rate/velocity
cross-terms during integral windup), not genuine tilt-driven translation --
because that mechanism was absent from both the report's model and ours.

**Fix** (`dynamics.py`): added the standard gravity-distribution terms
(rotating gravity from earth frame into body frame via R^T): `u_dot += g*
sin(theta)`, `v_dot += -g*sin(phi)*cos(theta)`, `w_dot` uses `-g*cos(phi)*
cos(theta)` instead of bare `-g`. Matches every mainstream flight-dynamics
text; not reproduced from the PDF (which lacks it).

**Consequence**: Table 1's manually-tuned outer x/y gains (k_px=k_py=0.2,
k_ix=k_iy=0.1, k_dx=k_dy=0.1 -- `ControllerGains`'s defaults, kept for
historical/Fig.-12 compatibility) were tuned against the old,
non-translating dynamics and are now **unstable**: verified diverging to
NaN by t~40s for the same 100s hover+small-disturbance scenario that
previously "worked" (i.e. drifted, per the old bug). Added
`gains_pole_placement_outer` (`controller.py`): the outer PID's own formula
(`k_p=wn^2/g, k_i=wn^3/g, k_d=2*zeta*wn/g`) exactly cancels the
now-correct plant's `g/s^2` relation (position from commanded tilt, given
the much-faster inner attitude loop), leaving closed-loop poles at
`s^3+2*zeta*wn*s^2+wn^2*s+wn^3=0`, independent of g -- a clean,
principled design. `wn_outer=0.6 rad/s, zeta_outer=1.0` (5x below the
inner loop's `wn=3rad/s`, needed for valid cascade separation) is verified
stable and accurate: a 1m step settles to <0.01% error by t=60s; tracking a
1m-radius circle at z=10m needs `omega` well below the outer loop's own
bandwidth (not just under the small-angle tilt limit -- `omega=0.5rad/s`,
comfortably small-angle, still gave ~65% tracking error since it exceeds
the loop's own bandwidth) -- `omega=0.125rad/s` (2 laps/100s) gives ~1-2%
tracking error.

Updated `reproduce_snider_figures.py`/`test_closed_loop_reproduction.py`
and `animate_flight_3d.py` to use `gains_pole_placement_outer` +
`integral_clamp=0.5`; they now correctly show bounded station-keeping
(settling back near the origin) for the small-disturbance hover scenario,
not Fig. 13's drift. Added `animate_flight_3d_circle.py`: a 100s, 2-lap,
1m-radius circle at z=10m, ~1-2% tracking error, `plots/
flight_animation_circle_100s.mp4`.

## 2026-09-03 — 100s 3D flight animation + integral anti-windup option

Added `uav_sim/validation_tests/animate_flight_3d.py`: a 3D animation of a
100s closed-loop flight (hover @ pz=0.5m, report's own small tilt
disturbance), saved to `plots/flight_animation_100s.mp4` (+ a filmstrip
PNG). Extending the existing 10s closed-loop validation
(`reproduce_snider_figures.py`) out to 100s surfaced a real finding:

**Listing 2's controller has no anti-windup**, matching the report exactly
(the firmware version, Listing 6 p.28, does add
`constrain(integral,-0.5,0.5)`, but the MATLAB "Simulation" controller used
for Fig. 12/13/Section 6 does not). Over 10s this doesn't matter, but
verified over 100s: with a persistent small steady-state position error
(matching Fig. 13's own described behavior), `integral_x`/`integral_y` grow
unboundedly, driving `phi`/`theta` to diverge (reached ~2 rad by t=100s,
well past the theta=90deg singularity) -- a real property of the report's
own controller architecture, not a bug in this implementation, just never
visible in the report's own 10s-long test.

Added an opt-in `integral_clamp` parameter to `CascadedController.__init__`
(default `None`, exactly reproducing Listing 2's behavior and every existing
test unchanged) so a caller can clamp `integral_x`/`integral_y` to
`+-integral_clamp` each step, matching the firmware's own anti-windup bound
(0.5). `animate_flight_3d.py` uses `integral_clamp=0.5` for a watchable
100s flight: `pz` reaches 0.5m by ~2s and holds, `px`/`py` drift slowly and
roughly linearly to ~+0.38m/-0.43m by t=100s (same behavior as the 10s
validation, just continued), and `phi`/`theta` stay bounded within
~+-0.14rad instead of diverging.

## 2026-09-03 — Stability-margin validation (src/stability_analysis.py)

Implemented Section 6's frequency-domain analysis (Eq. 14-19, all 6 loops:
Z, L inner, L outer, M inner, M outer, N) using the `control` package, and
compared computed gain/phase margins + crossover frequency against the
report's own stated Section 6 results (p.12-16) -- a third, frequency-domain
validation alongside the time-domain (`test_closed_loop_reproduction.py`)
and open-loop (`uav_sim/tests/`, `uav_sim/validation_tests/*.py`) checks.

**Result, after fixing one more report discrepancy (below): all 6 loops
match to within the report's own 3-sig-fig rounding** (gain margins to
0.00dB, phase margins to 0.02-0.3 degrees, crossover frequencies to
0.00-0.01 rad/s) -- locked in as `test_stability_margins.py`.

**One more report discrepancy found**: the printed Eq. 16/18 for the outer
loops reads `Lo_outer = g*(kdy*s^2+kpy*s+kiy)/(s*(1+s/N)) *
Lo_inner*(I+Lo_inner)^-1`, but Listing 3's own MATLAB (p.18) computes
`OLy = PID_tf*CLL/s` -- an **extra `/s`** beyond what the printed equation
shows. Without it, the computed L/M outer-loop crossover frequency comes out
~4x too high (~26-30 rad/s vs the stated ~7 rad/s) and the gain margin
comes out infinite instead of the stated finite ~7dB; with it, both match
almost exactly. `stability_analysis.py` implements the extra `/s` (i.e.
follows the code, which is what actually produced Fig. 8/10), documented
inline at both `phi_outer_open_loop` and `theta_outer_open_loop`.

## 2026-09-03 — Cascaded PID/PD controller (src/controller.py) + closed-loop validation

Implemented Snider's cascaded PID/PD controller architecture (Section 5,
Listing 2, p.10-11) as `uav_sim/src/controller.py`, then drove this
project's own plant with it, using the report's exact initial condition and
setpoint (Listing 4, p.21), to compare against Figures 12 and 13 as a
closed-loop validation of the dynamics model on top of the open-loop checks
already in `uav_sim/tests/` and `uav_sim/validation_tests/`.

**Result**: `uav_sim/validation_tests/reproduce_snider_figures.py` (run it
directly for plots+numbers; `test_closed_loop_reproduction.py` locks in the
qualitative properties as a regression test) reproduces both figures well:
`pz` rises and settles at 0.500m by ~2.5s with no overshoot (matches Fig. 12
closely), and `px`/`py` show a small (~0.03-0.04m over 10s), monotonic,
nonzero drift of the same sign pattern visible in Fig. 13 (px positive, py
negative) rather than either perfect station-keeping or runaway divergence —
matching the report's own description ("unsuccessful control of the px and
py drift ... unable to completely mitigate the steady-state error"). Note:
the exact drift *magnitude* in Fig. 13's own plot could not be verified
precisely (no raw data, only a rendered PDF image to eyeball at modest
resolution) — the match claimed here is qualitative (sign, shape, order of
magnitude, monotonicity), not an exact numeric overlay.

**Three more report discrepancies found while building this (added to the
running list from the bit-for-bit dynamics check):**

1. **Table 1's `k_pphi` "Simulation" entry (p.29) is an apparent
   digit-transposition typo.** Every other tabulated gain matches Listing
   2's own `zeta`/`w_n` formulas (`k_p = J*w_n^2`, `k_d = J*2*zeta*w_n`) to
   4-5 significant figures -- e.g. `k_dphi` computes to 1.19316e-4 vs
   tabulated 1.1932e-4 (match), `k_ptheta` to 1.499004e-4 vs tabulated
   1.499e-4 (match) -- but `k_pphi = Jxx*w_n_roll^2` with Listing 2's own
   `w_n_roll=3.0` computes to **1.4914539e-4**, not the tabulated
   **1.4194e-4** (digits "914" vs "194" transposed). `controller.py` computes
   all gains from Listing 2's zeta/w_n parameters directly (not copied from
   Table 1), so this typo is naturally avoided rather than reproduced.
2. **Section 5.1's prose formula disagrees with Listing 2's own code on a
   sign.** The prose (p.6) writes `L = ... + k_dphi*p_hat` (implying
   `+k_dphi*p`), but Listing 2's code (p.11) computes `e_p = 0 - p` then
   `L = k_pphi*e_phi + k_dphi*e_p` (i.e. `-k_dphi*p`). Implemented per the
   code, since that's what actually produced Fig. 12/13's plotted results.
3. **Listing 4's own step-response simulation (p.21-23) omits the motor lag
   entirely**, despite Section 5.3's block diagrams (Fig. 2-5) showing a
   `10/(s+10)` lag block between each controller output and the plant --
   Listing 4 calls the dynamics function directly on the raw controller
   output. `reproduce_snider_figures.py` runs both a "literal" variant (no
   lag, matching Listing 4 exactly) and a "full_plant" variant
   (`CrazyflieSim`, with the lag, matching this project's own designed
   architecture) side by side -- both reproduce Fig. 12/13 similarly well
   (final states agree to within ~0.005m of each other), confirming the lag
   is a second-order effect for this near-hover, slow-drift scenario.

## 2026-09-03 — Extended validation suite (uav_sim/validation_tests/)

Added a second, deeper layer of testing beyond `uav_sim/tests/`'s basic checks,
covering 6 distinct validation methods (each written by a separate agent,
working from exact numeric derivations/tolerances pre-verified by hand to
avoid handing off fragile or wrong test specs):

- `test_closed_form.py` — 3 tests against analytically-known solutions: exact
  free-fall parabola (bypassing the actuator lag), exact motor-lag
  exponential step response, and the Eq. 8-11 trim-control approximation
  (`pdot ~= L/Jxx` to ~0.11% relative error) — including an explicit check
  that a pure roll moment also produces a nonzero `rdot = gammas.g4*L` (a
  real Jxz cross-coupling effect, not a bug).
- `test_convergence.py` — confirms `plant.rk4_step` actually achieves 4th-order
  convergence via Richardson doubling (measured ratio 16.04 vs the
  theoretical 16).
- `test_conservation.py` — confirms torque-free rotation exactly conserves
  rotational kinetic energy and angular-momentum magnitude, to ~1e-15
  relative error. Uses the *aerospace* product-of-inertia sign convention
  (`J_eff` off-diagonal = `-Jxz`, not `+Jxz` as printed directly in Eq. 4) —
  verified symbolically with sympy that this is the exact tensor the
  report's own gamma formulas solve Euler's rigid-body equation for. Getting
  this sign wrong (my own first attempt did) produces a spurious ~9%/2s
  "energy drift" that looks like a bug but isn't one.
- `test_cross_validation.py` — cross-validates the project's hand-written
  RK4 against `scipy.integrate.solve_ivp` (DOP853) integrating the identical
  ODE: agreement to ~3.45e-10 over a 1s rollout. A gym-pybullet-drones
  comparison is stubbed as a documented `pytest.skip` (not attempted — would
  need patching that library's cf2x URDF mass/inertia to match Eq. 4's
  values, which differ substantially from its defaults).
- `visualize_trajectory.py` — standalone script (not a pytest test),
  produces `plots/altitude_step.png` and `plots/roll_doublet.png` plus
  printed PASS/FAIL sanity summaries. Found that a naive symmetric
  roll-moment doublet can never reverse `phi`'s sign, because `pdot = g3*L`
  is a pure double-integrator with no restoring torque for a pure-roll
  input — fixed by making the second-phase moment larger, not a plant bug.
- `test_fuzz.py` — 100 random-control episodes (fixed seed), asserts no
  NaN/Inf and bounded state, explicitly detecting and excluding the known
  theta=+-90deg singularity (7/100 episodes hit it) rather than either
  ignoring it or failing on it.

All 12 new tests pass (1 skip, by design). Full suite:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest uav_sim/tests/ uav_sim/validation_tests/ -v`
(the `PYTEST_DISABLE_PLUGIN_AUTOLOAD` env var works around an unrelated
globally-installed ROS pytest plugin that otherwise crashes on a missing
`yaml` module).

## 2026-09-03 — Bit-for-bit verification against Snider_Ian_ESE4481_Final.pdf

Did a full bit-for-bit check of every matrix/equation in `src/` against freshly
re-fetched, full-resolution pages of the source PDF (not just the earlier
transcription), including a numpy check of matrix identities the report itself
claims (e.g. that Eq. 22 is the inverse of Eq. 20).

### Confirmed exact matches

- Inertia matrix, Eq. 4 (`params.py` `J_full`) — all 6 distinct entries.
- Mass = 0.033 kg — cross-checked via Table 1's `k_pz = m*wn^2 = 0.033*3^2 =
  0.297`, which matches Table 1 exactly.
- Position kinematics, body-frame velocity dynamics, Euler-angle kinematics,
  and angular-rate equations (Section 2, p.2) — every term, sign, and
  coefficient in `dynamics.py` matches, including the R = Rz*Ry*Rx rotation
  (hand-verified by multiplying the three elementary matrices).
- Gamma formulas G1..G8 themselves (p.6 MATLAB `angular` function) — all 8
  match `params.py`'s `compute_gammas`.
- Motor lag H(s) = 10/(s+10), Eq. 13 -> tau=0.1s — matches `actuator.py`.
- Mixer, Eq. 20, and Mixer^-1, Eq. 22 — both match `mixer.py` exactly
  (re-verified at full zoom on PDF pages 25 and 26 of the file, printed pages
  24-25).

### Discrepancies found in the PDF itself (not in this code's favor to silently match)

1. **Gamma denominator typo, p.6 (PDF page 6), MATLAB `inertia_matrix(J)`
   function, line reading `gamma = Jx*Jy - Jxz^2`.** Every other gamma formula
   in the same report (G1, G3, G4, G7, G8) divides by this same `gamma`, and
   all of them are structurally identical to the standard aircraft-dynamics
   derivation (Beard & McLain-style), where the denominator is
   `Jx*Jz - Jxz^2`, not `Jx*Jy - Jxz^2`. Numerically these are not close:
   `Jx*Jy ~= 2.76e-10` vs `Jx*Jz ~= 4.85e-10` (a ~76% difference, since
   `Jzz` is nearly double `Jxx`/`Jyy`). Using the report's literal `Jx*Jy`
   would change every angular-rate coupling term (roll/pitch/yaw cross-terms)
   by that same ~76% factor.

   **Decision (2026-09-03): kept the corrected `Jxx*Jzz - Jxz**2` in
   `params.compute_gammas`.** It is physically standard, and it's the only
   choice consistent with the report's own other 7 gamma formulas. This is a
   deliberate, disclosed deviation from the PDF's literal text at this one
   line, not an error in this codebase.

2. **Mixer (Eq. 20) and Mixer^-1 (Eq. 22) are not actually inverses.**
   Verified numerically: the Eq. 20 matrix is **singular** (rank 3, det=0) —
   its 3rd row is exactly the negative of its 4th row
   (`[-1,1,-1,1] == -[1,-1,1,-1]`). So `Mixer @ Mixer_inv != I` for the
   matrices exactly as printed in Eq. 20/22. This is a bug in the report
   itself, unrelated to this code. `mixer.py` transcribes both matrices
   exactly as printed (a faithful transcription of a self-inconsistent
   source) and is reference-only — not wired into `CrazyflieSim` — so this
   inconsistency has no effect on simulated behavior.

3. **Minor, no action needed:** Listing 1 (p.4) substitutes `mass=33` (not
   `0.033`) when computing the numeric linearized `B` matrix for the
   controllability check — an isolated units bug in that one script. Not
   used anywhere else in the report (Table 1's gains confirm `mass=0.033` is
   the real, correct value used throughout), so it doesn't affect any
   parameter choice in this codebase.

### Prior note (same day, initial implementation)

- Caught and fixed a transcription error in `mixer.py`'s `MIXER_INV` during
  the initial planning pass (rows 3-4 had been transposed relative to the
  PDF's actual Eq. 22). Re-verified against the PDF a second time in this
  pass; now confirmed correct as printed.
