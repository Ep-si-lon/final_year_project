"""Originally reproduced Snider Figures 12/13 (p.20-21) bit-for-bit. NO
LONGER DOES, by design, since dynamics.py's translational EOM was corrected
(added the gravity-distribution terms the report's own printed equation is
missing -- see dynamics.py's DEVIATION docstring and CHANGELOG.md). Fig.
13's own px/py "drift" turned out to be an artifact of that missing
coupling, not genuine tilt-driven translation, so a corrected model
shouldn't reproduce it -- it should do BETTER (station-keeping, not
drifting).

This also required replacing the report's own outer x/y gains (Table 1's
manually-tuned k_px=k_py=0.2 etc, `ControllerGains`'s defaults) with
`gains_pole_placement_outer` (src/controller.py): those were tuned against
the pre-fix (non-translating) dynamics and are unstable once real
tilt-to-translate coupling exists (verified: this exact scenario, run with
the default gains post-fix, still stays bounded but oscillates far more).

Setup still matches Listing 4 (p.21) for pz/initial-condition/duration:
    ref  = [0, 0, 0.5, 0, 0, 0]   (hover at pz=0.5m)
    x0   = zeros(12) + [0,0,0, 0,0,0, 0.01,0.01,0.02, 0,0,0]  (small IC tilt)
    dt   = 1/500 = 0.002s, T = 10s

Two variants are run and plotted:
  1. "literal": rigid_body_deriv driven directly by the controller output,
     NO motor lag -- matches Listing 4's own script structurally (it also
     omits the motor lag, unlike Section 5.3's block diagrams -- see
     CHANGELOG.md), though the dynamics/gains inside no longer match.
  2. "full_plant": CrazyflieSim (RK4 + motor lag), i.e. this project's
     complete plant architecture as designed in uav_sim/src/plant.py.
Both use RK4 (this project's disclosed integrator choice) rather than
Listing 4's own forward Euler.

Not a pytest test (no exact numeric ground truth to assert against) -- run
directly: `python3 uav_sim/validation_tests/reproduce_snider_figures.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uav_sim.src.controller import CascadedController, gains_pole_placement_outer
from uav_sim.src.dynamics import rigid_body_deriv
from uav_sim.src.params import CrazyflieParams, compute_gammas
from uav_sim.src.plant import CrazyflieSim

REF6 = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
X0 = np.array([0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.02, 0, 0, 0], dtype=float)
DT = 1 / 500
T = 10.0
STEPS = int(T / DT)


def _rk4(f, x, u, dt, *args):
    k1 = f(x, u, *args)
    k2 = f(x + 0.5 * dt * k1, u, *args)
    k3 = f(x + 0.5 * dt * k2, u, *args)
    k4 = f(x + dt * k3, u, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def run_literal(params, gammas):
    """No motor lag: controller output feeds rigid_body_deriv directly."""
    gains = gains_pole_placement_outer(params)
    controller = CascadedController(params=params, gains=gains, dt=DT, integral_clamp=0.5)
    state = X0.copy()
    log = np.zeros((STEPS + 1, 12))
    log[0] = state
    for i in range(STEPS):
        u4 = controller.compute(state, REF6)
        state = _rk4(rigid_body_deriv, state, u4, DT, params, gammas)
        log[i + 1] = state
    return log


def run_full_plant(params):
    """This project's complete architecture: CrazyflieSim (RK4 + motor lag)."""
    gains = gains_pole_placement_outer(params)
    controller = CascadedController(params=params, gains=gains, dt=DT, integral_clamp=0.5)
    sim = CrazyflieSim(params=params, x0=X0.copy())
    log = np.zeros((STEPS + 1, 12))
    log[0] = sim.state
    for i in range(STEPS):
        u4 = controller.compute(sim.state, REF6)
        log[i + 1] = sim.step(u4, dt=DT)
    return log


def summarize(name, t, log):
    pz, px, py = log[:, 2], log[:, 0], log[:, 1]
    settle_idx = t >= 2.5
    settled_pz = pz[settle_idx]
    print(f"--- {name} ---")
    print(f"  pz(10s) = {pz[-1]:.4f} m (target 0.5m)")
    print(f"  pz settled (t>=2.5s) mean={settled_pz.mean():.4f}, "
          f"max={settled_pz.max():.4f}, min={settled_pz.min():.4f}")
    print(f"  px(10s) = {px[-1]:.4f} m, py(10s) = {py[-1]:.4f} m  "
          f"(post-fix: expect station-keeping near 0, NOT Fig. 13's drift -- "
          f"that drift was an artifact of the missing tilt-to-translate "
          f"coupling, now corrected)")
    print(f"  peak |px|={np.abs(px).max():.4f}m, peak |py|={np.abs(py).max():.4f}m")


def main():
    params = CrazyflieParams()
    gammas = compute_gammas(params)
    t = np.linspace(0, T, STEPS + 1)

    log_literal = run_literal(params, gammas)
    log_full = run_full_plant(params)

    summarize("literal (no motor lag, matches Listing 4)", t, log_literal)
    summarize("full_plant (CrazyflieSim, RK4 + motor lag)", t, log_full)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plots (numeric summary above still valid)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for log, label, style in [(log_literal, "literal", "-"), (log_full, "full_plant", "--")]:
        axes[0].plot(t, log[:, 2], style, label=label)
        axes[1].plot(t, log[:, 0], style, label=label)
        axes[2].plot(t, log[:, 1], style, label=label)

    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("pz [m]")
    axes[0].set_title("pz step response (still matches Fig. 12)")
    axes[0].legend()

    axes[1].set_ylabel("px [m]")
    axes[1].set_title("px response (post-fix: station-keeping, not Fig. 13's drift)")

    axes[2].set_ylabel("py [m]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title("py response (post-fix: station-keeping, not Fig. 13's drift)")

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()
    out_path = Path(__file__).parent / "plots" / "snider_reproduction.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
