# uav_trajectory_tracking -- assumption log and open questions

## Status: Stages 1-3 complete (2026-09-04)

Built and tested, in this order:
- `src/params.py`, `src/actuator.py`, `src/dynamics.py`, `src/plant.py` --
  self-contained fork of uav_sim's already-corrected nonlinear plant.
- `src/trajectory.py` -- general minimum-snap/minimum-acceleration
  piecewise-polynomial generator (KKT-solved QP). Bug found and fixed during
  development: internal waypoints' START position wasn't constrained (only
  each segment's END was), letting the QP leave it floating -- caught by
  `tests/test_trajectory.py::test_multisegment_hits_waypoints_exactly`.
- `src/flatness.py` -- differential-flatness feedforward (Z, phi, theta,
  psi, p, q, r) from a trajectory, Euler-angle-native (not SO(3)). Real bug
  found and fixed during development: the generic SO(3)-literature
  construction (project a heading vector, cross-product for the other body
  axes) does NOT exactly preserve commanded yaw once tilt is nonzero --
  replaced with an exact closed-form solve for (phi,theta) holding psi
  fixed, verified to machine precision (`tests/test_flatness.py`).
- `src/controller.py` -- cascaded feedback + flatness feedforward + output
  saturation. Closed-loop tracking verified on a step and a circle
  (`tests/test_controller.py`): RMS position error 1-4cm across all three
  validation trajectories.
- `src/reference_trajectories.py` -- the confirmed validation set (straight
  line, circle, helix), each built by fitting the minimum-snap generator
  through sampled waypoints (exercises the real generator, not a shortcut).
- `src/stability_analysis.py` -- linearized-per-loop Bode/margin/step
  analysis (5 loops: Z, phi, theta, psi, outer x/y), using `python-control`
  (new dependency, see requirements.txt -- justified by Q14's explicit ask
  for Bode/response plots). One real finding, not a bug: the outer x/y loop
  is "conditionally stable" (type-3 open loop) -- provably stable at the
  actual design gains via pole location + Routh-Hurwitz, but its raw
  single-crossing gain margin reads negative; documented in that module and
  in `tests/test_stability_analysis.py` rather than treated as a defect.
- `validation_tests/run_validation_suite.py` -- numeric summary + time-series
  plots + 3D animation for all 3 trajectories.
- `validation_tests/stability_analysis_report.py` -- Bode + step-response
  plots + margins table for all 5 loops.
- 15/15 unit tests passing (`tests/`); run with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest uav_trajectory_tracking/tests/`
  -- the env var works around an unrelated pre-existing issue (a ROS
  `launch_pytest` plugin auto-loads and fails on a missing `yaml` import;
  affects `uav_sim`'s own tests too, not something this project introduced).

## Post-delivery fix (2026-09-04, same day): circle/helix weren't actually circular

User inspection caught it: the first version's "circle"/"helix" validation
trajectories visibly weren't round. Root-caused to two compounding issues in
`src/reference_trajectories.py`, both fixed:

1. Only 12 waypoints/lap -- the minimum-snap fit only has to touch those 12
   points; nothing constrains the arcs BETWEEN them to follow the true
   circle, so it bowed in/out (measured: radius ranged 0.93-1.09 on a
   nominal r=1.0 circle). Fixed: 36+ waypoints/lap.
2. Rest-to-rest (zero velocity/accel/jerk) boundary condition at the
   seam -- physically wrong for a continuous loop (real circular motion
   never stops), and it produced a visible outward spike right at the
   start/end point. Fixed: feed the ANALYTIC uniform-circular-motion
   velocity/acceleration/jerk in as the boundary condition instead (exact
   and loop-consistent for integer n_laps, since start/end angle then
   coincide mod 2pi).

After the fix: radius holds to 1e-9 m over the whole loop (was 0.93-1.09).
This surfaced a secondary, separate issue in the VALIDATION HARNESS (not
the trajectory itself): `run_validation_suite.py` and
`tests/test_controller.py` started the simulated plant from REST at the
trajectory's start position, but the corrected trajectory now starts
already in motion (tangential velocity) -- an avoidable initial velocity-
mismatch transient that inflated tracking-error numbers (circle RMS
0.14m) without reflecting an actual controller problem. Fixed by seeding
the sim's initial velocity from the trajectory's own start velocity too;
circle RMS is back down to 0.05m. All validation plots/animations and unit
tests re-run and re-verified after both fixes.

## Independent audit (2026-09-04, same day): re-derivation from first principles, not a re-read

Prompted by "how much have you hallucinated" -- every non-trivial math claim
below was re-derived from scratch (symbolically via sympy, or numerically
against the actual simulated plant), independent of the code and independent
of this conversation's own earlier derivations, specifically so a wrong
claim couldn't just be re-confirming itself.

CONFIRMED CORRECT (full re-derivation matched code exactly):
- All 8 Gamma inertia-reduction coefficients (g1-g8, params.py): re-derived
  from Euler's rigid-body rotational equation of motion by hand, three
  times, until the aerospace product-of-inertia sign convention (tensor
  entry = -Jxz, per Beard & McLain) was used correctly -- first attempt
  (raw +Jxz convention) gave a REAL, non-matching discrepancy in g6's sign;
  switching to the aerospace convention resolved it exactly, for all 8
  coefficients (p_dot, q_dot, r_dot equations all matched term-for-term).
- Gravity-in-body-frame terms (dynamics.py's uvw_dot): re-derived
  g_body = R^T @ [0,0,-g] fully symbolically via sympy (not hand algebra
  this time) -- exact match.
- euler_rate_matrix: verified against the actual PHYSICAL DEFINITION of
  body angular velocity (dR/dt = R @ skew(p,q,r)), not just against its own
  inverse -- max error 1.6e-10 over 2000 random trials.
- Minimum-snap closed-form point-to-point oracle (the 35/-84/70/-20
  polynomial used in test_trajectory.py): re-derived independently via
  sympy Lagrange multipliers (not assumed from memory) -- exact match.
- Trajectory QP's cost matrix Q: confirmed positive semi-definite
  (eigenvalues >= 0) for several segment durations -- a precondition for
  the KKT solution to genuinely be a constrained MINIMUM rather than a
  saddle point (the closed-form match above is the stronger proof of this,
  but PSD-ness was checked directly too).
- controller.py's phi_c sign convention: verified analytically against
  dynamics.py's own coupling term (-g*sinphi*cos(theta) in v_dot) --
  positive y-position error must produce NEGATIVE commanded roll, which is
  what the code's `phi_c = ff.phi - (...)` does.

FOUND AND FIXED (a real gap, not a hallucination in the sense of a false
claim, but an incompleteness that would have overstated results to a
professor):
- `src/stability_analysis.py`'s 4 PD loops (Z, phi, theta, psi) originally
  linearized ONLY the rigid-body relation (1/(J*s^2)), omitting
  actuator.py's real first-order motor lag (tau_motor=0.1s) that sits
  between the controller's commanded output and what the rigid body
  actually receives. Caught by directly simulating dynamics.py's
  rigid_body_deriv under a small constant moment with the lag bypassed
  (matched the no-lag prediction to 1.3e-8) vs. through the real
  plant.py/CrazyflieSim WITH the lag (did NOT match) -- confirming the lag
  is a real, present pole the original analysis silently dropped.
  Quantified impact at the phi-loop's own crossover frequency: ~36 degrees
  of omitted phase lag. Fixed by putting H(s)=10/(s+10) in series with each
  PD loop's plant. Reported phase margins dropped from 76-84deg to a more
  honest 46-51deg (still stable, still a legitimate margin -- >45deg is the
  classical "good" threshold -- but the ORIGINAL numbers were optimistic
  and would not have withstood scrutiny). All plots/tests re-run after the
  fix.

STILL AN OPEN ASSUMPTION, not verified either way (flagging rather than
claiming): the outer x/y position loop's P(s)=g/s^2 model assumes the inner
attitude loop "is much faster and tracks its command perfectly." Given the
motor-lag finding above shows the inner loop's own margins are less
comfortable than first thought (46deg, not 80deg), this fast-inner-loop
assumption's accuracy has not been independently re-examined post-fix --
it is a documented ASSUMPTION carried into the outer-loop analysis, not a
verified fact. Flagging for anyone pressure-testing this further.

## Deferred, not forgotten (open follow-up items)

- **Hardware deployment (Q2 answer: yes, eventually).** Current saturation
  limits (Z in [0,2mg], L/M in +-5e-4 N*m, N in +-3e-4 N*m) are the same
  "physically reasonable, not measured" placeholders `uav_sim`'s own
  `test_fuzz.py` uses -- NOT real Crazyflie motor limits, and there is no
  PWM mixer (force/torque -> per-motor commands) yet. Both are needed before
  any real hardware attempt; deferred per Q17 ("test at the end," sim-first).
- SO(3)/geometric (Lee/Leok/McClamroch) controller: not built. Only
  revisit if a future trajectory needs tilt beyond what Euler angles handle
  well (per D2/Q3's "gentle/indoor" scope decision).
- Tracking-error pass/fail metric (Q15) was never explicitly answered;
  RMS + max + ITAE are reported (default), no formal pass/fail threshold
  was requested or set beyond the unit tests' own assertions.

Per the working rule: nothing gets built until every assumption below is
either confirmed or overridden, and every open question has an answer.
This file is the single source of truth for that; update it as decisions
get made, don't let decisions live only in chat history.

## Important correction before anything else

A lot of the research material this project is based on (Part C's
assumption table, "Part D" fixes) describes limitations of **the Snider
PDF's own printed equations** -- hover-only gravity shortcut, hover-linearized
attitude kinematics, etc. Those specific fixes are **already implemented**
in `uav_sim/src/dynamics.py`, in this same repo:

- Gravity is already rotated into the body frame using the *full* rotation
  matrix at the current attitude (`g_body = R^T @ [0,0,-g]`), not the PDF's
  literal "gravity only in body-z" shortcut. This is mathematically
  equivalent to the "rotate thrust into earth frame, leave gravity in earth
  frame" form Part D1 recommends -- same physics, expressed in the other
  frame. It is valid at *any* attitude, not just near-hover.
- Attitude kinematics already use the full nonlinear Euler-rate matrix
  (not the hover-linearized identity), so `dot(phi),dot(theta),dot(psi)`
  are correct away from hover too (up to the theta=+-90 singularity, which
  is intrinsic to Euler angles, not a bug).
- Rotational dynamics already use the standard Gamma-reduction (matches
  Beard & McLain's aircraft-dynamics form), RK4 integration, and a
  first-order motor lag.

So `uav_trajectory_tracking` is **not** starting from the PDF's broken
hover-only model -- it can start from `uav_sim`'s already-general nonlinear
model. The open design question is not "is the plant broken," it's "do we
reuse this plant as-is, or does trajectory-tracking's fidelity/attitude-
representation needs justify a different model" -- see Q1/Q2 below.

## Defaults I will use unless told otherwise

These are reasonable engineering defaults, not guesses -- but every one is
overridable, and I'd rather list a default and be corrected than silently
assume.

| # | Topic | Default |
|---|---|---|
| D1 | Location/independence | New top-level folder `uav_trajectory_tracking/` (sibling to `uav_sim/`, done), importing `uav_sim.src.{params,dynamics,plant}` rather than forking copies -- one physical model, not two to keep in sync |
| D2 | Attitude representation | Keep Euler angles (full nonlinear, already in `uav_sim`), not quaternion/SO(3), *unless* target trajectories need near-90-degree pitch or inversions |
| D3 | Controller | Extend the existing cascaded feedback controller to accept a time-varying reference, plus add differential-flatness feedforward on top (Mellinger & Kumar) -- not a full SE(3) geometric controller, unless aggressive/large-angle maneuvers are a stated goal |
| D4 | Trajectory generator | Minimum-snap polynomial trajectories through waypoints (Mellinger & Kumar / Richter-Bry-Roy style), outputting position through snap, rather than only hand-coded circle/figure-8 parametric curves |
| D5 | Fidelity | Keep the Gamma-reduction inertia model and no rotor/aero drag (matches `uav_sim`, matches the original flatness derivation) -- revisit only if target speeds are high |
| D6 | Motor model | Keep the existing first-order lag (`tau=0.1s`) as-is |
| D7 | Saturation/mixer | Add explicit thrust/torque saturation limits to the controller output (per Greiff's point about saturation-induced loss of authority), but do NOT add a per-motor PWM mixer unless hardware deployment is a goal (matches `uav_sim`'s existing scope: force/torque interface, not PWM) |
| D8 | Path planning | Out of scope -- trajectory *tracking* only; trajectories are specified as waypoints+times you provide, no RRT*/obstacle avoidance |
| D9 | Validation deliverables | Numeric tracking-error summary + time-series plots + 3D animation (matching `uav_sim/validation_tests`' existing style), for at least: a moving hover-to-hover step, a circle, and a figure-8 |
| D10 | Testing | pytest unit tests for the trajectory generator and feedforward math (closed-form checks), validation-script-style (not pytest) for full closed-loop animations -- matching `uav_sim`'s existing split between `tests/` and `validation_tests/` |
| D11 | Build order | Incremental: (1) trajectory generator + flatness feedforward, checked open-loop against known closed-form inputs; (2) feedback added on top, tracking error validated on a step and a circle; (3) stop and check in before deciding whether SO(3)/geometric control (Part D above) is actually needed |
| D12 | Stack | Pure Python/numpy/matplotlib, same as `uav_sim` -- no ROS/CasADi/Drake unless a specific need for constrained optimization (e.g. Richter-Bry-Roy's corridor constraints) comes up |

## Decisions locked in (answered via AskUserQuestion + inline --answer- edits)

| # | Decision |
|---|---|
| Envelope | Gentle/indoor (<2 m/s, moderate tilt) -- resolves Q3/Q9/Q10/Q11 together |
| Plant | Self-contained fork (Q1) -- done: `src/params.py`, `src/actuator.py`, `src/dynamics.py` forked from `uav_sim` |
| Attitude representation | Euler angles, full nonlinear (Q3, follows from envelope) |
| Controller | Cascaded feedback + differential-flatness feedforward (Q4), NOT full SE(3) geometric |
| Trajectory generator | Minimum-snap polynomial through waypoints (Q6) -- done: `src/trajectory.py`, validated in `tests/test_trajectory.py` |
| Feedforward derivative order | Full snap for x/y/z, accel for yaw (Q5) -- done: `src/flatness.py`, validated in `tests/test_flatness.py` |
| Fidelity | Gamma-reduction inertia, no rotor/aero drag (Q9/Q10) |
| Path planning | Out of scope, confirmed (Q8) |
| **Hardware deployment** | **Confirmed goal (Q2): simulate first, then deploy to real Crazyflie hardware.** This changes D7 -- a per-motor PWM mixer with real min/max allocation is NOT optional long-term, but is deferred until after closed-loop sim tracking is validated (matches Q17's "test at the end" -- no reason to build hardware-facing code before the sim-only math is proven). Flagging now so it isn't forgotten: **open item for a later stage, not dropped.** |
| Validation trajectory set | Straight line, circle, helix (circle + climbing z), at minimum (Q7) |
| Validation deliverables | Numeric summary + plots + animation, PLUS **stability analysis: Bode plots and step-response plots** (Q14) -- added for the loops that admit a clean linearization (Z-loop, attitude inner loop): both are PD-controlled 2nd-order systems, meaningful to Bode-plot; the outer position loop's linearization is the pole-placement design's own closed-loop poles, also plottable. Full nonlinear closed-loop trajectory tracking itself isn't a single LTI system, so "Bode plot of the whole thing" isn't meaningful -- the stability analysis targets each linearizable sub-loop, not the full nonlinear system. Using `scipy.signal` (already installed, no new dependency).
| Tracking-error metric | Not explicitly answered (Q15) -- default: RMS position error + max error + ITAE (matches `uav_sim`'s own PSO fitness convention, so results are comparable) |
| Saturation | Not explicitly overridden (Q12) -- default: yes, explicit thrust/torque saturation added to controller output |
| Build order / check-ins | Confirmed as proposed (Q16): generator+feedforward, then feedback, then stop before SO(3) is considered. Check in **at the end only** (Q17), not after each stage -- proceeding straight through. |

## Open questions -- need your answer before I write any dynamics/controller/generator code
### (superseded by the table above where answered; left below for full traceability)

### Scope and integration
1. Reuse `uav_sim`'s `CrazyflieParams`/`dynamics.py`/`plant.py` as a dependency (import them), or should `uav_trajectory_tracking` be fully self-contained with its own copy of the plant? (Default: reuse -- D1.)
2. Is this meant to stay a pure-simulation study (like `uav_sim`), or is hardware deployment on a real Crazyflie eventually a goal? This changes whether saturation/mixer/PWM work (Q13-14) actually matters now or can be deferred.   "--answer-it is meant to deploy in the real hardware after testing in the simulation."

### Attitude representation (Part 1 of your checklist)
3. Euler angles (full nonlinear, matching `uav_sim`) good enough for your target trajectories, or do you specifically want quaternion/SO(3) from the start? What's the most aggressive maneuver you actually intend to test -- e.g. is a 90-degree-plus pitch, a loop, or a recovery-from-inverted ever in scope? (Default: Euler, D2 -- but this is the single highest-leverage question, since it reshapes every equation downstream.)

### Controller design
4. Cascaded feedback + differential-flatness feedforward (D3's default), or do you specifically want the Lee/Leok/McClamroch SE(3) geometric controller implemented (possibly as a second variant for comparison, like davidegorbani's repo does three variants side by side)?
5. If feedforward is in scope: how far up the derivative chain do you want the trajectory generator to go -- acceleration only (enough for a basic feedforward thrust/attitude estimate) or full snap (needed for Mellinger & Kumar's exact algebraic inversion)? (Default: full snap, since it's what the theory actually needs and it's not much extra work once you're doing polynomial trajectories.)

### Trajectory generation
6. Minimum-snap polynomial through arbitrary waypoints+times (D4), or do you want to start smaller with hand-coded parametric trajectories (circle, figure-8/lemniscate, straight-line-with-smoothed-corners) and only build the general minimum-snap optimizer once those are validated?
7. Do you have specific test trajectories already in mind (radius, speed, duration, number of waypoints), or should I propose a first validation set (e.g. 1m-radius circle at a few speeds, a lemniscate, a 4-waypoint minimum-snap path)? ""--answer-please create a validation set of trajectories to use for validation purposes like straight line.. circle helical etc."
8. Path planning (obstacle avoidance, RRT*) -- confirmed out of scope (D8), or is that actually wanted eventually? --answer-"No obstacle avoidance is not required"

### Fidelity
9. Confirm: keep the Gamma-reduction (symmetric-inertia-approximation) rotational dynamics rather than the full asymmetric inertia matrix? (Default: yes, D5 -- only matters at high angular rates.)
10. Confirm: neglect rotor/aerodynamic drag, matching both `uav_sim` and the original Mellinger & Kumar flatness derivation? Or is "fast" flight (where Faessler et al.'s drag-augmented flatness would matter) actually a target regime?
11. What's the actual target flight envelope in one sentence -- e.g. "indoor, <2 m/s, gentle waypoint following" vs "aggressive racing-style maneuvers"? This single answer resolves Q3, Q9, Q10 and D5/D2 all at once, so if you only answer one question, answer this one.

### Actuator/practical constraints
12. Do you want explicit min/max thrust and per-axis torque saturation added to the controller output? (Default: yes, D7.)
13. Per-motor PWM mixer with individual motor min/max and a real 4-in/4-out allocation matrix -- needed only if hardware deployment (Q2) is a goal. Confirm out of scope for now?

### Validation and deliverables
14. Confirm the validation deliverable set (numeric summary + plots + 3D animation, D9) -- anything to add or drop? No enough.. give a stability analysis as well like boded plots and responce plots 
15. Any specific tracking-error metric you want reported (e.g. RMS position error, max error, ITAE like `uav_sim`'s PSO fitness) as the primary pass/fail criterion?

### Process
16. Confirm the incremental build order (D11) -- generator+feedforward first, feedback second, SO(3)/geometric as a later decision point -- or do you want a different sequencing (e.g. controller first against hand-coded trajectories, generator second)? --answer-"No same is enough"
17. How much do you want checked in with you between stages -- a review after each of the 3 stages in D11, or just at the end? --answer-"Let's test in the ends..."
