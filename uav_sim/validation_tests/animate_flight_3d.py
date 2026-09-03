"""3D animation of a 100s closed-loop flight: CascadedController driving
CrazyflieSim, hovering at pz=0.5m with the report's own small initial-tilt
disturbance (matching reproduce_snider_figures.py's setup, extended from
10s to 100s).

Uses `gains_pole_placement_outer` for the x/y outer loop, NOT
`ControllerGains`'s defaults (Table 1's manually-tuned 0.2/0.1/0.1) --
those were tuned against dynamics.py's pre-fix translational equation
(no tilt-to-translate coupling at all) and now diverge to NaN by t~40s
once real coupling exists (verified). See dynamics.py's DEVIATION
docstring, controller.py's docstring, and CHANGELOG.md.

Also uses `integral_clamp=0.5` (matching the firmware's own anti-windup
bound, Listing 6 p.28): Listing 2's controller as literally specified has
no anti-windup, which over 100s lets phi/theta grow via integral windup
whenever a steady-state error persists -- a real property of the report's
own controller (see controller.py's docstring), independent of the
dynamics fix.

Not a pytest test -- run directly: `python3
uav_sim/validation_tests/animate_flight_3d.py`. Needs matplotlib + ffmpeg
(falls back to a GIF via Pillow if ffmpeg isn't available).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uav_sim.src.controller import CascadedController, gains_pole_placement_outer
from uav_sim.src.dynamics import rotation_body_to_earth
from uav_sim.src.params import CrazyflieParams
from uav_sim.src.plant import CrazyflieSim

REF6 = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
X0 = np.array([0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.02, 0, 0, 0], dtype=float)
DT = 0.002
T = 100.0
STEPS = int(T / DT)
FRAME_STRIDE = 100  # every 0.2s of sim time -> 500 frames for a 20s/25fps video
ARM_LENGTH = 0.12   # visually exaggerated (real Crazyflie arm ~0.045m) for visibility


def simulate():
    params = CrazyflieParams()
    gains = gains_pole_placement_outer(params)
    controller = CascadedController(params=params, gains=gains, dt=DT, integral_clamp=0.5)
    sim = CrazyflieSim(params=params, x0=X0.copy())
    log = np.zeros((STEPS + 1, 12))
    log[0] = sim.state
    for i in range(STEPS):
        u4 = controller.compute(sim.state, REF6)
        log[i + 1] = sim.step(u4, dt=DT)
    return log


def quad_arm_points(px, py, pz, phi, theta, psi, arm_length=ARM_LENGTH):
    """4 rotor-arm tips in earth frame, X-configuration (Fig. 14a)."""
    R = rotation_body_to_earth(phi, theta, psi)
    body_pts = arm_length * np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    ]) / np.sqrt(2)
    earth_pts = (R @ body_pts.T).T + np.array([px, py, pz])
    return earth_pts


def main():
    print(f"Simulating {T:.0f}s ({STEPS} steps at dt={DT})...")
    log = simulate()
    px, py, pz = log[:, 0], log[:, 1], log[:, 2]
    phi, theta, psi = log[:, 6], log[:, 7], log[:, 8]

    print(f"pz(100s)={pz[-1]:.4f}m, px(100s)={px[-1]:.4f}m, py(100s)={py[-1]:.4f}m")
    print(f"peak |phi|={np.abs(phi).max():.4f}rad, peak |theta|={np.abs(theta).max():.4f}rad "
          f"(bounded thanks to integral_clamp=0.5)")

    frame_idx = np.arange(0, STEPS + 1, FRAME_STRIDE)
    t_frames = frame_idx * DT

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib not installed -- cannot animate (numeric summary above still valid)")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    pad = 0.15
    ax.set_xlim(px.min() - pad, px.max() + pad)
    ax.set_ylim(py.min() - pad, py.max() + pad)
    ax.set_zlim(0, pz.max() + pad)
    ax.set_xlabel("px [m]")
    ax.set_ylabel("py [m]")
    ax.set_zlabel("pz [m]")
    ax.set_title("100s closed-loop flight (hover @ 0.5m + small tilt disturbance)")

    trail_line, = ax.plot([], [], [], "b-", linewidth=1, alpha=0.6)
    arm1, = ax.plot([], [], [], "r-", linewidth=2)
    arm2, = ax.plot([], [], [], "r-", linewidth=2)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame_i):
        i = frame_idx[frame_i]
        trail_line.set_data(px[:i + 1], py[:i + 1])
        trail_line.set_3d_properties(pz[:i + 1])

        pts = quad_arm_points(px[i], py[i], pz[i], phi[i], theta[i], psi[i])
        arm1.set_data([pts[0, 0], pts[3, 0]], [pts[0, 1], pts[3, 1]])
        arm1.set_3d_properties([pts[0, 2], pts[3, 2]])
        arm2.set_data([pts[1, 0], pts[2, 0]], [pts[1, 1], pts[2, 1]])
        arm2.set_3d_properties([pts[1, 2], pts[2, 2]])

        time_text.set_text(f"t = {t_frames[frame_i]:.1f}s")
        return trail_line, arm1, arm2, time_text

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=40, blit=False)

    plots_dir = Path(__file__).parent / "plots"
    if shutil.which("ffmpeg"):
        out_path = plots_dir / "flight_animation_100s.mp4"
        anim.save(out_path, writer=FFMpegWriter(fps=25))
    else:
        out_path = plots_dir / "flight_animation_100s.gif"
        anim.save(out_path, writer=PillowWriter(fps=25))
    print(f"Saved animation: {out_path}")
    plt.close(fig)

    # filmstrip: static snapshots at a few times, for a quick look without playing a video
    snap_times = [0, 2, 10, 25, 50, 75, 100]
    fig2 = plt.figure(figsize=(18, 3))
    for k, st in enumerate(snap_times):
        i = min(int(st / DT), STEPS)
        ax2 = fig2.add_subplot(1, len(snap_times), k + 1, projection="3d")
        ax2.set_xlim(px.min() - pad, px.max() + pad)
        ax2.set_ylim(py.min() - pad, py.max() + pad)
        ax2.set_zlim(0, pz.max() + pad)
        ax2.plot(px[:i + 1], py[:i + 1], pz[:i + 1], "b-", linewidth=1, alpha=0.6)
        pts = quad_arm_points(px[i], py[i], pz[i], phi[i], theta[i], psi[i])
        ax2.plot([pts[0, 0], pts[3, 0]], [pts[0, 1], pts[3, 1]], [pts[0, 2], pts[3, 2]], "r-", linewidth=2)
        ax2.plot([pts[1, 0], pts[2, 0]], [pts[1, 1], pts[2, 1]], [pts[1, 2], pts[2, 2]], "r-", linewidth=2)
        ax2.set_title(f"t={st}s")
    fig2.tight_layout()
    filmstrip_path = plots_dir / "flight_filmstrip_100s.png"
    fig2.savefig(filmstrip_path, dpi=100)
    print(f"Saved filmstrip: {filmstrip_path}")


if __name__ == "__main__":
    main()
