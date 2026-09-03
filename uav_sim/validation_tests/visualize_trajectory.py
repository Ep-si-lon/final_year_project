"""Manual visual/sanity-check script for the Crazyflie plant model.

Not a pytest test -- run directly:
    python3 uav_sim/validation_tests/visualize_trajectory.py

Produces two PNGs under uav_sim/validation_tests/plots/ so a human can
eyeball the plant's behavior for a couple of representative maneuvers, and
prints plain pass/fail-style sanity summaries to stdout.

Maneuver 1 (altitude step) reuses the exact throwaway Z-only PD controller
from uav_sim/tests/test_step_response.py, driving the plant with the
report's own gains (K_PZ=0.297, K_DZ=0.198) toward a 0.5m altitude target.

Maneuver 2 (roll doublet) is open-loop: a small constant roll moment L is
commanded positive, then negative, then zero, each for 1s, with Z held at
hover thrust, to check the sign convention and general shape of the roll
response.
"""
import os
import sys

import numpy as np

# Allow running directly as `python3 uav_sim/validation_tests/visualize_trajectory.py`
# from the repo root, where the repo root isn't on sys.path by default (unlike
# under pytest, which adds it automatically).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from uav_sim.src.plant import CrazyflieSim

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

K_PZ = 0.297
K_DZ = 0.198


def run_altitude_step():
    """Reproduces the throwaway Z-only PD controller from
    uav_sim/tests/test_step_response.py, logging pz(t) and w(t)."""
    sim = CrazyflieSim()
    dt = 0.002
    duration = 10.0
    steps = int(duration / dt)

    t_log = np.zeros(steps)
    pz_log = np.zeros(steps)
    w_log = np.zeros(steps)

    for i in range(steps):
        pz, vz = sim.state[2], sim.state[5]  # w == earth-frame vz at zero tilt
        Z = sim.params.mass * sim.params.g + K_PZ * (0.5 - pz) + K_DZ * (0.0 - vz)
        state = sim.step(np.array([Z, 0.0, 0.0, 0.0]), dt=dt)
        t_log[i] = (i + 1) * dt
        pz_log[i] = state[2]
        w_log[i] = state[5]

    return t_log, pz_log, w_log


def run_roll_doublet():
    """Open-loop roll doublet: +L for 1s, -L for 1s, 0 for 1s, Z held at
    hover thrust throughout. Logs phi(t) and p(t).

    Tuning note (deviation from the literal 3e-4 magnitude suggested in the
    task, disclosed here): the roll axis of a hovering quadrotor has no
    restoring torque (Euler's equation here reduces to p_dot = g3*L exactly,
    a pure double integrator, since q=r=0 throughout a pure-roll maneuver).
    A perfectly symmetric +L/-L doublet of equal magnitude and duration
    exactly cancels the accumulated roll rate at the end of the second
    phase and phi merely plateaus -- it can never cross back through zero,
    regardless of how L_MAG is scaled up or down. Two changes were needed
    to get a physically meaningful, non-blowing-up sign-convention check:
    (1) L_MAG=3e-4 was far too large -- it drives phi through multiple full
    rotations within 1s (peak |phi| ~18 rad), so the magnitude was scaled
    down by ~2-3 orders of magnitude; (2) the second-phase magnitude is
    deliberately ~4x the first-phase magnitude (still same duration, 1.0s
    each) so the reversed moment overshoots the exact-cancellation point
    and genuinely drives phi negative during the second phase, rather than
    merely decelerating it back toward zero.
    """
    sim = CrazyflieSim()
    dt = 0.002
    phase_duration = 1.0
    total_duration = 3.0 * phase_duration
    steps = int(total_duration / dt)

    L_MAG_1 = 1e-6   # phase 1: +L
    L_MAG_2 = 4e-6   # phase 2: -L, deliberately larger -- see note above
    hover_Z = sim.params.mass * sim.params.g

    t_log = np.zeros(steps)
    phi_log = np.zeros(steps)
    p_log = np.zeros(steps)

    for i in range(steps):
        t = i * dt
        if t < phase_duration:
            L = L_MAG_1
        elif t < 2 * phase_duration:
            L = -L_MAG_2
        else:
            L = 0.0
        state = sim.step(np.array([hover_Z, L, 0.0, 0.0]), dt=dt)
        t_log[i] = (i + 1) * dt
        phi_log[i] = state[6]
        p_log[i] = state[9]

    return t_log, phi_log, p_log


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not available -- install it with "
            "'pip install matplotlib' to generate plots. Exiting gracefully."
        )
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # Maneuver 1: altitude step response
    # ---------------------------------------------------------------
    print("=" * 60)
    print("Maneuver 1: Altitude step response (Snider Fig. 12 reproduction)")
    print("=" * 60)

    t1, pz_log, w_log = run_altitude_step()

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1a.plot(t1, pz_log, label="pz(t)")
    ax1a.axhline(0.5, color="red", linestyle="--", label="target pz=0.5m")
    ax1a.set_ylabel("Altitude pz [m]")
    ax1a.set_title("Altitude step response -- Snider Fig. 12 reproduction")
    ax1a.legend()
    ax1a.grid(True)

    ax1b.plot(t1, w_log, color="tab:orange", label="w(t) (vertical velocity)")
    ax1b.set_xlabel("Time [s]")
    ax1b.set_ylabel("Vertical velocity w [m/s]")
    ax1b.legend()
    ax1b.grid(True)

    fig1.tight_layout()
    altitude_path = os.path.join(PLOTS_DIR, "altitude_step.png")
    fig1.savefig(altitude_path, dpi=150)
    plt.close(fig1)
    print(f"Saved plot: {altitude_path}")

    final_pz = pz_log[-1]
    settled = abs(final_pz - 0.5) < 0.02
    print(f"pz at t=10s = {final_pz:.4f}m (target 0.5m)")
    if settled:
        print("Altitude settled near 0.5m at t=10s: PASS")
    else:
        print("Altitude settled near 0.5m at t=10s: FAIL")

    max_pz = pz_log.max()
    no_overshoot = max_pz <= 0.6
    print(f"Peak pz reached = {max_pz:.4f}m")
    if no_overshoot:
        print("No excessive overshoot (peak pz <= 0.6m): PASS")
    else:
        print("No excessive overshoot (peak pz <= 0.6m): FAIL")

    # ---------------------------------------------------------------
    # Maneuver 2: roll doublet response
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("Maneuver 2: Roll doublet response")
    print("=" * 60)

    t2, phi_log, p_log = run_roll_doublet()

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax2a.plot(t2, phi_log, label="phi(t) (roll angle)")
    ax2a.axvline(1.0, color="gray", linestyle="--")
    ax2a.axvline(2.0, color="gray", linestyle="--")
    ax2a.set_ylabel("Roll angle phi [rad]")
    ax2a.set_title("Roll doublet response")
    ax2a.legend()
    ax2a.grid(True)

    ax2b.plot(t2, p_log, color="tab:orange", label="p(t) (roll rate)")
    ax2b.axvline(1.0, color="gray", linestyle="--")
    ax2b.axvline(2.0, color="gray", linestyle="--")
    ax2b.set_xlabel("Time [s]")
    ax2b.set_ylabel("Roll rate p [rad/s]")
    ax2b.legend()
    ax2b.grid(True)

    fig2.tight_layout()
    roll_path = os.path.join(PLOTS_DIR, "roll_doublet.png")
    fig2.savefig(roll_path, dpi=150)
    plt.close(fig2)
    print(f"Saved plot: {roll_path}")

    phase1_mask = t2 < 1.0
    phase2_mask = (t2 >= 1.0) & (t2 < 2.0)
    went_positive = np.any(phi_log[phase1_mask] > 0)
    went_negative = np.any(phi_log[phase2_mask] < 0)
    sign_ok = went_positive and went_negative
    if sign_ok:
        print(
            "phi went positive during phase 1 (+L) and negative during "
            "phase 2 (-L): PASS"
        )
    else:
        print(
            "phi went positive during phase 1 (+L) and negative during "
            "phase 2 (-L): FAIL"
        )

    peak_abs_phi = np.max(np.abs(phi_log))
    print(f"Peak |phi| reached = {peak_abs_phi:.6f} rad")


if __name__ == "__main__":
    main()
