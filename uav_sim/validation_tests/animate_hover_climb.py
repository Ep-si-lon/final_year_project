"""3D animation of a closed-loop climb: x0 = zeros(12) (on the ground, level,
at rest -- everything 0), reference ref6 = [0, 0, 0.5, 0, 0, 0] (target
pz=0.5m, station-keep px=py=0). Unlike animate_hover_zero.py's open-loop
constant-thrust case, here the CascadedController reads the state every
step and commands whatever [Z,L,M,N] drives it toward the reference -- so
the drone is EXPECTED to actually climb from pz=0 to pz=0.5m and settle
there, not sit still.

Uses gains_pole_placement_outer (see controller.py's docstring) -- the
Table 1 manually-tuned gains diverge once the dynamics fix's real
tilt-to-translate coupling is present.

Not a pytest test -- run directly: `python3
uav_sim/validation_tests/animate_hover_climb.py`. Needs matplotlib + ffmpeg
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

REF6 = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])  # target: pz=0.5m, px=py=0
DT = 0.002
T = 15.0
STEPS = int(T / DT)
FRAME_STRIDE = 30  # every 0.06s of sim time -> 250 frames at 25fps = 10s video
ARM_LENGTH = 0.12  # visually exaggerated (real Crazyflie arm ~0.045m) for visibility


def simulate():
    params = CrazyflieParams()
    gains = gains_pole_placement_outer(params)
    controller = CascadedController(params=params, gains=gains, dt=DT, integral_clamp=0.5)
    sim = CrazyflieSim(params=params, x0=np.zeros(12))  # ground, level, at rest
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
    print(f"Simulating {T:.0f}s ({STEPS} steps at dt={DT}), x0=zeros(12), ref=pz=0.5m...")
    log = simulate()
    px, py, pz = log[:, 0], log[:, 1], log[:, 2]
    phi, theta, psi = log[:, 6], log[:, 7], log[:, 8]

    print(f"pz(0s)={pz[0]:.4f}m -> pz({T:.0f}s)={pz[-1]:.4f}m (target 0.5m)")
    print(f"final px={px[-1]:.4f}m, py={py[-1]:.4f}m (target 0)")
    print(f"peak |phi|={np.abs(phi).max():.4f}rad, peak |theta|={np.abs(theta).max():.4f}rad")

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
    ax.set_title("Closed-loop climb: x0=0 (ground) -> ref pz=0.5m")

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
        out_path = plots_dir / "flight_animation_hover_climb.mp4"
        anim.save(out_path, writer=FFMpegWriter(fps=25))
    else:
        out_path = plots_dir / "flight_animation_hover_climb.gif"
        anim.save(out_path, writer=PillowWriter(fps=25))
    print(f"Saved animation: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
