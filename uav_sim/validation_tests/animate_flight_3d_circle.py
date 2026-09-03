"""3D animation of a closed-loop flight tracking a circular reference
trajectory: radius=1m, altitude z=10m, angular rate omega=0.5 rad/s
(~12.6s/lap, ~8 laps over the 100s run). Mirrors animate_flight_3d.py's
structure but swaps the fixed hover setpoint for src/reference.circle().

omega=0.5 keeps the required centripetal tilt small (~1.5deg for radius=1m,
tilt ~= omega^2*radius/g) so the tracking stays within the controller's
small-angle design assumption (Section 5.3's "sin(phi)~=phi" trim relation).
Starts the drone already at the circle's t=0 point (radius,0,z) at rest, so
the only transient is spinning up to the commanded tangential velocity.

Not a pytest test -- run directly: `python3
uav_sim/validation_tests/animate_flight_3d_circle.py`. Needs matplotlib +
ffmpeg (falls back to GIF via Pillow if ffmpeg isn't available).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uav_sim.src.controller import CascadedController
from uav_sim.src.params import CrazyflieParams
from uav_sim.src.plant import CrazyflieSim
from uav_sim.src.reference import circle
from uav_sim.validation_tests.animate_flight_3d import quad_arm_points

RADIUS = 1.0
Z = 10.0
OMEGA = 0.5
DT = 0.002
T = 100.0
STEPS = int(T / DT)
FRAME_STRIDE = 100  # every 0.2s of sim time -> 500 frames for a 20s/25fps video
ARM_LENGTH = 0.18


def simulate():
    params = CrazyflieParams()
    controller = CascadedController(params=params, dt=DT, integral_clamp=0.5)
    x0 = np.array([RADIUS, 0, Z, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    sim = CrazyflieSim(params=params, x0=x0)
    log = np.zeros((STEPS + 1, 12))
    ref_log = np.zeros((STEPS + 1, 6))
    log[0] = sim.state
    ref_log[0] = circle(0.0, RADIUS, Z, OMEGA)
    for i in range(STEPS):
        t = i * DT
        ref6 = circle(t, RADIUS, Z, OMEGA)
        ref_log[i] = ref6
        u4 = controller.compute(sim.state, ref6)
        log[i + 1] = sim.step(u4, dt=DT)
    ref_log[-1] = circle(STEPS * DT, RADIUS, Z, OMEGA)
    return log, ref_log


def main():
    print(f"Simulating {T:.0f}s circle (radius={RADIUS}m, z={Z}m, omega={OMEGA}rad/s, "
          f"{T*OMEGA/(2*np.pi):.2f} laps)...")
    log, ref_log = simulate()
    px, py, pz = log[:, 0], log[:, 1], log[:, 2]
    phi, theta, psi = log[:, 6], log[:, 7], log[:, 8]

    tracking_err = np.linalg.norm(log[:, :3] - ref_log[:, :3], axis=1)
    print(f"tracking error: mean={tracking_err.mean():.4f}m, max={tracking_err.max():.4f}m "
          f"(after initial transient, t>10s: mean={tracking_err[int(10/DT):].mean():.4f}m)")
    print(f"peak |phi|={np.abs(phi).max():.4f}rad ({np.degrees(np.abs(phi).max()):.2f}deg), "
          f"peak |theta|={np.abs(theta).max():.4f}rad ({np.degrees(np.abs(theta).max()):.2f}deg)")
    print(f"pz range: [{pz.min():.4f}, {pz.max():.4f}] (target {Z}m)")

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

    ring_theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(RADIUS * np.cos(ring_theta), RADIUS * np.sin(ring_theta),
             np.full_like(ring_theta, Z), "k--", linewidth=1, alpha=0.4, label="reference circle")

    pad = 0.3
    ax.set_xlim(-RADIUS - pad, RADIUS + pad)
    ax.set_ylim(-RADIUS - pad, RADIUS + pad)
    ax.set_zlim(Z - pad, Z + pad)
    ax.set_xlabel("px [m]")
    ax.set_ylabel("py [m]")
    ax.set_zlabel("pz [m]")
    ax.set_title(f"Circle tracking: r={RADIUS}m, z={Z}m, {T:.0f}s ({T*OMEGA/(2*np.pi):.1f} laps)")
    ax.legend(loc="upper right")

    trail_line, = ax.plot([], [], [], "b-", linewidth=1, alpha=0.6)
    arm1, = ax.plot([], [], [], "r-", linewidth=2)
    arm2, = ax.plot([], [], [], "r-", linewidth=2)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame_i):
        i = frame_idx[frame_i]
        trail_line.set_data(px[:i + 1], py[:i + 1])
        trail_line.set_3d_properties(pz[:i + 1])

        pts = quad_arm_points(px[i], py[i], pz[i], phi[i], theta[i], psi[i], arm_length=ARM_LENGTH)
        arm1.set_data([pts[0, 0], pts[3, 0]], [pts[0, 1], pts[3, 1]])
        arm1.set_3d_properties([pts[0, 2], pts[3, 2]])
        arm2.set_data([pts[1, 0], pts[2, 0]], [pts[1, 1], pts[2, 1]])
        arm2.set_3d_properties([pts[1, 2], pts[2, 2]])

        time_text.set_text(f"t = {t_frames[frame_i]:.1f}s")
        return trail_line, arm1, arm2, time_text

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=40, blit=False)

    plots_dir = Path(__file__).parent / "plots"
    if shutil.which("ffmpeg"):
        out_path = plots_dir / "flight_animation_circle_100s.mp4"
        anim.save(out_path, writer=FFMpegWriter(fps=25))
    else:
        out_path = plots_dir / "flight_animation_circle_100s.gif"
        anim.save(out_path, writer=PillowWriter(fps=25))
    print(f"Saved animation: {out_path}")
    plt.close(fig)

    snap_times = [0, 3, 15, 30, 50, 75, 100]
    fig2 = plt.figure(figsize=(18, 3))
    for k, st in enumerate(snap_times):
        i = min(int(st / DT), STEPS)
        ax2 = fig2.add_subplot(1, len(snap_times), k + 1, projection="3d")
        ax2.set_xlim(-RADIUS - pad, RADIUS + pad)
        ax2.set_ylim(-RADIUS - pad, RADIUS + pad)
        ax2.set_zlim(Z - pad, Z + pad)
        ax2.plot(RADIUS * np.cos(ring_theta), RADIUS * np.sin(ring_theta),
                  np.full_like(ring_theta, Z), "k--", linewidth=1, alpha=0.4)
        ax2.plot(px[:i + 1], py[:i + 1], pz[:i + 1], "b-", linewidth=1, alpha=0.6)
        pts = quad_arm_points(px[i], py[i], pz[i], phi[i], theta[i], psi[i], arm_length=ARM_LENGTH)
        ax2.plot([pts[0, 0], pts[3, 0]], [pts[0, 1], pts[3, 1]], [pts[0, 2], pts[3, 2]], "r-", linewidth=2)
        ax2.plot([pts[1, 0], pts[2, 0]], [pts[1, 1], pts[2, 1]], [pts[1, 2], pts[2, 2]], "r-", linewidth=2)
        ax2.set_title(f"t={st}s")
    fig2.tight_layout()
    filmstrip_path = plots_dir / "flight_filmstrip_circle_100s.png"
    fig2.savefig(filmstrip_path, dpi=100)
    print(f"Saved filmstrip: {filmstrip_path}")


if __name__ == "__main__":
    main()
