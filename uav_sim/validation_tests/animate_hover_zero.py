"""3D animation of the pure zero-IC hover equilibrium: x0 = [0,0,0.5,0,...,0]
(pz=0.5m, everything else zero), constant commanded thrust = m*g (no
controller, no reference trajectory -- exactly test_hover.py's setup, just
visualized at a nonzero altitude).

Per dynamics.py, none of the state derivatives depend on pz itself (the
open-loop plant is translation-invariant), so this is still an exact fixed
point -- the commanded hover thrust exactly cancels gravity regardless of
altitude, and all state derivatives vanish (see conversation trace). This
animation is EXPECTED to show a perfectly motionless quadrotor sitting at
pz=0.5m for the whole run -- that's the point: it's a visual confirmation
that hover is a fixed point of the plant at any altitude, not a bug in the
animation.

Not a pytest test -- run directly: `python3
uav_sim/validation_tests/animate_hover_zero.py`. Needs matplotlib + ffmpeg
(falls back to a GIF via Pillow if ffmpeg isn't available).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uav_sim.src.dynamics import rotation_body_to_earth
from uav_sim.src.params import CrazyflieParams
from uav_sim.src.plant import CrazyflieSim

DT = 0.002
T = 5.0
STEPS = int(T / DT)
FRAME_STRIDE = 20  # every 0.04s of sim time -> 125 frames at 25fps = 5s video
ARM_LENGTH = 0.12  # visually exaggerated (real Crazyflie arm ~0.045m) for visibility


def simulate():
    params = CrazyflieParams()
    x0 = np.zeros(12)
    x0[2] = 0.5  # pz = 0.5m, everything else 0
    sim = CrazyflieSim(params=params, x0=x0)
    hover_cmd = np.array([params.mass * params.g, 0.0, 0.0, 0.0])
    log = np.zeros((STEPS + 1, 12))
    log[0] = sim.state
    for i in range(STEPS):
        log[i + 1] = sim.step(hover_cmd, dt=DT)
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
    print(f"Simulating {T:.0f}s ({STEPS} steps at dt={DT}), x0=[0,0,0.5,0,...,0], cmd=hover thrust...")
    log = simulate()
    px, py, pz = log[:, 0], log[:, 1], log[:, 2]
    phi, theta, psi = log[:, 6], log[:, 7], log[:, 8]

    non_pz = np.delete(log, 2, axis=1)  # every state column except pz should stay exactly 0
    print(f"max |state except pz| over the whole run = {np.abs(non_pz).max():.2e} "
          f"(should be ~0 -- exact equilibrium)")
    print(f"final state: px={px[-1]:.6f} py={py[-1]:.6f} pz={pz[-1]:.6f}")

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
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_zlim(0.5 - pad, 0.5 + pad)
    ax.set_xlabel("px [m]")
    ax.set_ylabel("py [m]")
    ax.set_zlabel("pz [m]")
    ax.set_title("Hover equilibrium at pz=0.5m, rest=0: cmd=m*g (expect: no motion)")

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

        time_text.set_text(f"t = {t_frames[frame_i]:.2f}s")
        return trail_line, arm1, arm2, time_text

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=40, blit=False)

    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    if shutil.which("ffmpeg"):
        out_path = plots_dir / "flight_animation_hover_pz05.mp4"
        anim.save(out_path, writer=FFMpegWriter(fps=25))
    else:
        out_path = plots_dir / "flight_animation_hover_pz05.gif"
        anim.save(out_path, writer=PillowWriter(fps=25))
    print(f"Saved animation: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
