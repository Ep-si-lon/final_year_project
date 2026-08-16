"""Particle Swarm Optimization search for the outer x/y position-loop gains.

Three separate searches, run independently so their results can be compared:

  "3var"        outer kp/ki/kd only -- inner attitude loop held fixed at the
                margin_safe formula values (wn=3.0, zeta=1.2). This is the
                same search space hand_tuned's grid search covered manually.
  "5var_formula" wn_inner, zeta_inner (shared symmetrically across roll and
                pitch, pole-placement formula) + outer kp/ki/kd -- mirrors
                the reference report's Sec 5.2 decomposition
                (reference_replica's parameterization), letting PSO pick the
                inner loop bandwidth/damping too instead of fixing it.
  "5var_raw"    kp_inner, kd_inner as raw torque-domain gains (bypassing the
                wn/zeta formula entirely, via controller.py's inner_gains
                override) + outer kp/ki/kd -- a more literal "5 free numbers"
                search with no pole-placement structure assumed at all.

All three share one gain vector across x and y (matching every existing
preset -- aggressive/margin_safe/hand_tuned/reference_replica are all
symmetric), per user's stated preference.

Fitness (combined approach): cheap frequency-domain margin check first
(stability_analysis.py's GM/PM targets, same >6dB/>45deg as the reference
report). Particles that fail get a penalty proportional to how badly they
fail -- steep enough that PSO reliably moves out of the infeasible region,
but still smooth (no simulation run at all for these, since a doomed
particle's time-domain error isn't informative). Particles that pass get
simulated (full circle-tracking run) and scored by ITAE (integral of
time-weighted absolute error) over the WHOLE run, not just a late
steady-state window -- see _tracking_error's docstring for why: a
last-N-seconds-mean fitness previously let PSO trade settling speed for a
marginally smaller final error (one run settled in ~70s instead of ~10s
for a ~2x tighter final number), because it had no visibility into, or
penalty for, how long the transient took. ITAE punishes error that lingers
late into the run much more than the same error early on, directly
penalizing slow settling and lightly-damped/ringing responses.
"""

import numpy as np

from dynamics import CrazyflieParams
from controller import CascadedPIDController
from stability_analysis import build_loops_from_controller, GM_TARGET_DB, PM_TARGET_DEG
import simulate

# Fixed inner-loop values for the "3var" search (= margin_safe's).
FIXED_WN_ATTITUDE = 3.0
FIXED_ZETA_ATTITUDE = 1.2

# Everything else (Z, yaw) is held at margin_safe's values in all three
# searches -- those loops aren't in question, only the outer x/y loop
# (and, in the 5-var searches, the inner roll/pitch loop that feeds it).
FIXED_KWARGS = dict(wn_z=3.0, zeta_z=1.0, wn_yaw=3.0, zeta_yaw=1.2)

SIM_DURATION = 100.0     # circle-tracking run, seconds, per particle evaluation

PENALTY_BASE = 1000.0


def _margin_violation(gm_db, pm_deg):
    """0 if this loop clears both targets, else how far it misses by."""
    gm_ok = (not np.isfinite(gm_db)) or gm_db > GM_TARGET_DB
    pm_ok = pm_deg > PM_TARGET_DEG
    if gm_ok and pm_ok:
        return 0.0
    gm_miss = 0.0 if gm_ok else (GM_TARGET_DB - gm_db)
    pm_miss = 0.0 if pm_ok else (PM_TARGET_DEG - pm_deg)
    return gm_miss + pm_miss


def _check_margins(ctrl, loop_names):
    """Sum of margin violations across the given loops (0 = all pass)."""
    import control
    loops = build_loops_from_controller(ctrl)
    total = 0.0
    for name in loop_names:
        gm, pm, _, _ = control.margin(loops[name])
        gm_db = 20 * np.log10(gm) if np.isfinite(gm) and gm > 0 else np.inf
        total += _margin_violation(gm_db, pm)
    return total


def _tracking_error(ctrl_kwargs):
    """ITAE (Integral of Time-weighted Absolute Error) over the FULL run,
    not just a late steady-state window. Weighting error by time t means an
    error that's still present late in the run costs far more than the same
    error size early on (when some transient is expected) -- this directly
    penalizes slow settling and lightly-damped/oscillatory responses (error
    that lingers or rings for a long time racks up a large time-weighted
    sum), not just wherever the trajectory happens to be at the end.
    A last-20s-mean fitness (the previous approach) is blind to *how* a
    candidate got to its final error -- a gain set that takes 70s to ring
    down to a tiny error scores identically to one that settles in 10s, as
    long as both land in the same place by the time the window starts. ITAE
    scores the whole trajectory, so the slow-ringing candidate is penalized
    for every second it spent still oscillating.
    Normalized by duration**2 to keep the scale in the same ballpark as a
    plain mean error (so PENALTY_BASE=1000 stays a safe separation from any
    feasible run's score) -- trapz(t*|e|, t) grows with duration**2 for a
    fixed error profile, so dividing by it makes scores roughly
    duration-independent."""
    log = simulate.run(duration=SIM_DURATION, trajectory="circle",
                        gain_preset=None, controller_kwargs=ctrl_kwargs)
    err = np.linalg.norm(log["err"], axis=1)
    t = log["t"]
    itae = np.trapezoid(t * err, t)
    return itae / SIM_DURATION ** 2


def _fitness_3var(x):
    kp, ki, kd = x
    ctrl_kwargs = dict(FIXED_KWARGS, wn_attitude=FIXED_WN_ATTITUDE, zeta_attitude=FIXED_ZETA_ATTITUDE,
                        outer_gains=dict(kp=kp, ki=ki, kd=kd))
    ctrl = CascadedPIDController(CrazyflieParams(), **ctrl_kwargs)
    violation = _check_margins(ctrl, ["y outer (via roll)", "x outer (via pitch)"])
    if violation > 0:
        return PENALTY_BASE + violation
    return _tracking_error(ctrl_kwargs)


def _fitness_5var_formula(x):
    wn_in, zeta_in, kp, ki, kd = x
    ctrl_kwargs = dict(FIXED_KWARGS, wn_attitude=wn_in, zeta_attitude=zeta_in,
                        outer_gains=dict(kp=kp, ki=ki, kd=kd))
    ctrl = CascadedPIDController(CrazyflieParams(), **ctrl_kwargs)
    violation = _check_margins(ctrl, ["phi inner (roll rate/angle)", "theta inner (pitch rate/angle)",
                                       "y outer (via roll)", "x outer (via pitch)"])
    if violation > 0:
        return PENALTY_BASE + violation
    return _tracking_error(ctrl_kwargs)


def _fitness_5var_raw(x):
    kp_in, kd_in, kp, ki, kd = x
    ctrl_kwargs = dict(FIXED_KWARGS, inner_gains=dict(kp=kp_in, kd=kd_in),
                        outer_gains=dict(kp=kp, ki=ki, kd=kd))
    ctrl = CascadedPIDController(CrazyflieParams(), **ctrl_kwargs)
    violation = _check_margins(ctrl, ["phi inner (roll rate/angle)", "theta inner (pitch rate/angle)",
                                       "y outer (via roll)", "x outer (via pitch)"])
    if violation > 0:
        return PENALTY_BASE + violation
    return _tracking_error(ctrl_kwargs)


SEARCHES = {
    "3var": dict(
        fitness=_fitness_3var,
        names=["kp_outer", "ki_outer", "kd_outer"],
        # Widened from [0,1]/[0,0.5]/[0,2] -- the previous run pinned
        # kd_outer exactly at 2.0 and kp_outer near 1.0, a sign the true
        # optimum sits past the old wall.
        bounds=np.array([[0.0, 3.0], [0.0, 0.5], [0.0, 6.0]]),
    ),
    "5var_formula": dict(
        fitness=_fitness_5var_formula,
        names=["wn_inner", "zeta_inner", "kp_outer", "ki_outer", "kd_outer"],
        bounds=np.array([[1.0, 15.0], [0.5, 2.0], [0.0, 1.0], [0.0, 0.5], [0.0, 2.0]]),
    ),
    "5var_raw": dict(
        fitness=_fitness_5var_raw,
        names=["kp_inner", "kd_inner", "kp_outer", "ki_outer", "kd_outer"],
        # kp_inner/kd_inner bounds unchanged (previous best sat well inside
        # them); kp_outer/kd_outer widened -- kp_outer pinned at 1.0 before.
        bounds=np.array([[1e-5, 5e-3], [1e-5, 2e-3], [0.0, 3.0], [0.0, 0.5], [0.0, 6.0]]),
    ),
}


def pso(fitness, bounds, n_particles=30, n_iters=40, w=0.7, c1=1.5, c2=1.5, seed=0, log=None):
    """Standard PSO: minimize `fitness` over the box defined by `bounds`
    (shape (n_dims, 2)). Returns (best_x, best_fitness, history) where
    history[i] is the best fitness found by the end of iteration i."""
    rng = np.random.default_rng(seed)
    n_dims = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]

    pos = rng.uniform(lo, hi, size=(n_particles, n_dims))
    vel = rng.uniform(-(hi - lo), (hi - lo), size=(n_particles, n_dims)) * 0.1

    pbest_pos = pos.copy()
    pbest_val = np.array([fitness(p) for p in pos])
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    history = [gbest_val]
    for it in range(n_iters):
        r1 = rng.uniform(size=(n_particles, n_dims))
        r2 = rng.uniform(size=(n_particles, n_dims))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        new_pos = pos + vel

        # Absorbing boundary: zero out the velocity component that pushed a
        # particle past a wall, instead of just clipping position and
        # leaving velocity untouched. Without this, a particle whose true
        # pull is past the wall gets teleported back to the wall every
        # iteration but keeps re-commanding "go further" next iteration too
        # -- it sits vibrating on the boundary forever, which looks exactly
        # like "the optimum is at the boundary" even when it isn't.
        hit_lo = new_pos < lo
        hit_hi = new_pos > hi
        vel[hit_lo | hit_hi] = 0.0
        pos = np.clip(new_pos, lo, hi)

        vals = np.array([fitness(p) for p in pos])
        improved = vals < pbest_val
        pbest_pos[improved] = pos[improved]
        pbest_val[improved] = vals[improved]

        it_best_idx = np.argmin(pbest_val)
        if pbest_val[it_best_idx] < gbest_val:
            gbest_val = pbest_val[it_best_idx]
            gbest_pos = pbest_pos[it_best_idx].copy()

        history.append(gbest_val)
        if log is not None:
            log(f"  iter {it + 1:3d}/{n_iters}: best fitness = {gbest_val:.5f}")

    return gbest_pos, gbest_val, history


def run_search(name, n_particles=30, n_iters=40, seed=0):
    spec = SEARCHES[name]
    print(f"\n=== PSO search: {name} ({len(spec['names'])} variables) ===")
    best_x, best_val, history = pso(
        spec["fitness"], spec["bounds"], n_particles=n_particles, n_iters=n_iters,
        seed=seed, log=print,
    )
    print(f"best: {dict(zip(spec['names'], best_x))}")
    print(f"fitness (ITAE, time-weighted error integral; or {PENALTY_BASE}+violation if infeasible): {best_val:.5f}")
    return best_x, best_val, history


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(SEARCHES.keys())
    results = {}
    for name in names:
        results[name] = run_search(name)

    print("\n=== summary ===")
    for name, (best_x, best_val, _) in results.items():
        print(f"{name:14s} fitness={best_val:.5f}  {dict(zip(SEARCHES[name]['names'], best_x))}")
