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
simulated (short circle-tracking run) and scored by steady-state tracking
error -- this is the real number that distinguishes stable presets from
each other (0.14m hand_tuned vs 0.20m margin_safe, etc).
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

SIM_DURATION = 15.0   # short circle-tracking run, seconds
SIM_STEADY_WINDOW = 5.0  # score = mean error over the last N seconds

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
    log = simulate.run(duration=SIM_DURATION, trajectory="circle",
                        gain_preset=None, controller_kwargs=ctrl_kwargs)
    err = np.linalg.norm(log["err"], axis=1)
    return err[log["t"] > (SIM_DURATION - SIM_STEADY_WINDOW)].mean()


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
        bounds=np.array([[0.0, 1.0], [0.0, 0.5], [0.0, 2.0]]),
    ),
    "5var_formula": dict(
        fitness=_fitness_5var_formula,
        names=["wn_inner", "zeta_inner", "kp_outer", "ki_outer", "kd_outer"],
        bounds=np.array([[1.0, 15.0], [0.5, 2.0], [0.0, 1.0], [0.0, 0.5], [0.0, 2.0]]),
    ),
    "5var_raw": dict(
        fitness=_fitness_5var_raw,
        names=["kp_inner", "kd_inner", "kp_outer", "ki_outer", "kd_outer"],
        bounds=np.array([[1e-5, 5e-3], [1e-5, 2e-3], [0.0, 1.0], [0.0, 0.5], [0.0, 2.0]]),
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
        pos = np.clip(pos + vel, lo, hi)

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
    print(f"fitness (steady-state tracking error, m; or {PENALTY_BASE}+violation if infeasible): {best_val:.5f}")
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
