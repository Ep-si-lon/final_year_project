"""Reproduces Snider Figures 12 and 13 (p.20-21) using this project's own
plant + the CascadedController (src/controller.py), as a closed-loop
validation of BOTH the controller implementation and the dynamics model:
if driving our own dynamics with the report's own controller/gains/initial
condition reproduces the report's own described closed-loop behavior, that's
strong evidence the plant matches the report beyond the open-loop checks in
uav_sim/tests/ and uav_sim/validation_tests/.

Setup matches Listing 4 (p.21) exactly:
    ref  = [0, 0, 0.5, 0, 0, 0]   (hover at pz=0.5m)
    x0   = zeros(12) + [0,0,0, 0,0,0, 0.01,0.01,0.02, 0,0,0]  (small IC tilt)
    dt   = 1/500 = 0.002s, T = 10s

Two variants are run and plotted:
  1. "literal": rigid_body_deriv driven directly by the controller output,
     NO motor lag -- this matches Listing 4's own script, which (despite
     Section 5.3's block diagrams showing a 10/(s+10) motor-lag block after
     the controller) has no such block; it calls the dynamics function
     directly on the raw controller output. See CHANGELOG.md.
  2. "full_plant": CrazyflieSim (RK4 + motor lag), i.e. this project's
     complete plant architecture as designed in uav_sim/src/plant.py.
Both use RK4 (this project's disclosed integrator choice) rather than
Listing 4's own forward Euler.

Not a pytest test (no exact numeric ground truth to assert against, only a
plot in the PDF to eyeball) -- run directly: `python3
uav_sim/validation_tests/reproduce_snider_figures.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uav_sim.src.controller import CascadedController
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
    controller = CascadedController(params=params, dt=DT)
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
    controller = CascadedController(params=params, dt=DT)
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
          f"(Fig. 13: small nonzero drift, order 0.01-0.5m, not exactly zero)")
    print(f"  px monotonic drift: {'yes' if np.all(np.diff(px) >= -1e-6) or np.all(np.diff(px) <= 1e-6) else 'no'}")


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
    axes[0].set_title("Reproduction of Snider Fig. 12 (pz step response)")
    axes[0].legend()

    axes[1].set_ylabel("px [m]")
    axes[1].set_title("Reproduction of Snider Fig. 13a (px drift)")

    axes[2].set_ylabel("py [m]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title("Reproduction of Snider Fig. 13b (py drift)")

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()
    out_path = Path(__file__).parent / "plots" / "snider_reproduction.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
