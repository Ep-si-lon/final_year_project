"""D9/Q7/Q14 validation deliverable: for each of the confirmed validation
trajectories (straight line, circle, helix), run the closed-loop
feedback+feedforward controller against the self-contained plant, and
produce (1) a numeric tracking-error summary, (2) time-series plots
(position/attitude/error), (3) a 3D animation.

Not a pytest test -- run directly:
    python3 uav_trajectory_tracking/validation_tests/run_validation_suite.py
Needs matplotlib + ffmpeg (falls back to GIF via Pillow if ffmpeg missing).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uav_trajectory_tracking.src.controller import (
    TrajectoryTrackingController, gains_pole_placement_outer,
)
from uav_trajectory_tracking.src.dynamics import rotation_body_to_earth
from uav_trajectory_tracking.src.params import CrazyflieParams
from uav_trajectory_tracking.src.plant import CrazyflieSim
from uav_trajectory_tracking.src.reference_trajectories import (
    circle_trajectory, helix_trajectory, straight_line_trajectory,
)

DT = 0.002
PLOTS_DIR = Path(__file__).parent / "plots"
ARM_LENGTH = 0.12


def simulate(traj, x0=None, dt=DT):
    params = CrazyflieParams()
    gains = gains_pole_placement_outer(params)
    controller = TrajectoryTrackingController(params=params, gains=gains, dt=dt, integral_clamp=0.5)
    if x0 is None:
        x0 = np.zeros(12)
        pos_vel0 = traj.eval_pos(traj.t_start, max_deriv=1)
        x0[:3] = pos_vel0[0]
        # Body velocity == earth velocity here since attitude starts at 0
        # (R=identity) -- matches the trajectory's own start-in-motion
        # boundary condition (circle/helix no longer start from rest, see
        # reference_trajectories.py); starting the SIM at rest while the
        # reference starts at full tangential speed would just measure an
        # avoidable velocity-tracking transient, not a real tracking result.
        x0[3:6] = pos_vel0[1]
    sim = CrazyflieSim(params=params, x0=x0)

    n_steps = int(round((traj.t_end - traj.t_start) / dt))
    t_log = np.zeros(n_steps + 1)
    state_log = np.zeros((n_steps + 1, 12))
    pos_des_log = np.zeros((n_steps + 1, 3))
    yaw_des_log = np.zeros(n_steps + 1)
    u4_log = np.zeros((n_steps, 4))

    state_log[0] = sim.state
    t_log[0] = traj.t_start
    pos_des_log[0] = traj.eval_pos(traj.t_start, max_deriv=0)[0]
    yaw_des_log[0] = traj.eval_yaw(traj.t_start, max_deriv=0)[0]

    for i in range(n_steps):
        t = traj.t_start + i * dt
        u4 = controller.compute(sim.state, traj, t)
        u4_log[i] = u4
        state_log[i + 1] = sim.step(u4, dt=dt)
        t_next = t + dt
        t_log[i + 1] = t_next
        pos_des_log[i + 1] = traj.eval_pos(t_next, max_deriv=0)[0]
        yaw_des_log[i + 1] = traj.eval_yaw(t_next, max_deriv=0)[0]

    return {
        "t": t_log, "state": state_log, "pos_des": pos_des_log,
        "yaw_des": yaw_des_log, "u4": u4_log, "params": params,
    }


def compute_metrics(log) -> dict:
    pos = log["state"][:, :3]
    pos_des = log["pos_des"]
    err = pos - pos_des
    err_norm = np.linalg.norm(err, axis=1)
    t = log["t"]

    rms = float(np.sqrt(np.mean(err_norm ** 2)))
    max_err = float(err_norm.max())
    itae = float(np.trapezoid(t * err_norm, t))
    final_err = float(err_norm[-1])

    return {
        "rms_pos_error_m": rms,
        "max_pos_error_m": max_err,
        "itae": itae,
        "final_pos_error_m": final_err,
    }


def make_plots(name: str, log: dict):
    import matplotlib.pyplot as plt

    t = log["t"]
    state = log["state"]
    pos, pos_des = state[:, :3], log["pos_des"]
    phi, theta, psi = state[:, 6], state[:, 7], state[:, 8]
    err = np.linalg.norm(pos - pos_des, axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    labels = ["px [m]", "py [m]", "pz [m]"]
    for i in range(3):
        axes[0].plot(t, pos_des[:, i], "--", label=f"{labels[i]} desired", alpha=0.7)
    axes[0].set_prop_cycle(None)
    for i in range(3):
        axes[0].plot(t, pos[:, i], label=labels[i])
    axes[0].set_ylabel("position [m]")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title(f"{name}: position tracking")

    axes[1].plot(t, np.degrees(phi), label="phi (roll)")
    axes[1].plot(t, np.degrees(theta), label="theta (pitch)")
    axes[1].plot(t, np.degrees(psi), label="psi (yaw)")
    axes[1].plot(t, np.degrees(log["yaw_des"]), "k--", label="psi desired", alpha=0.6)
    axes[1].set_ylabel("attitude [deg]")
    axes[1].legend(ncol=4, fontsize=8)

    axes[2].plot(t, err)
    axes[2].set_ylabel("|position error| [m]")
    axes[2].axhline(0, color="gray", linewidth=0.5)

    axes[3].plot(t[:-1], log["u4"][:, 0], label="Z [N]")
    ax3b = axes[3].twinx()
    ax3b.plot(t[:-1], log["u4"][:, 1] * 1e3, "r-", label="L [mN*m]", alpha=0.7)
    ax3b.plot(t[:-1], log["u4"][:, 2] * 1e3, "g-", label="M [mN*m]", alpha=0.7)
    ax3b.plot(t[:-1], log["u4"][:, 3] * 1e3, "b-", label="N [mN*m]", alpha=0.7)
    axes[3].set_ylabel("Z [N]")
    ax3b.set_ylabel("moments [mN*m]")
    axes[3].set_xlabel("t [s]")
    lines1, labs1 = axes[3].get_legend_handles_labels()
    lines2, labs2 = ax3b.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labs1 + labs2, ncol=4, fontsize=8)

    fig.tight_layout()
    out = PLOTS_DIR / f"{name}_timeseries.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def quad_arm_points(px, py, pz, phi, theta, psi, arm_length=ARM_LENGTH):
    R = rotation_body_to_earth(phi, theta, psi)
    body_pts = arm_length * np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    ]) / np.sqrt(2)
    return (R @ body_pts.T).T + np.array([px, py, pz])


def make_animation(name: str, log: dict, frame_stride: int = 25):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    t = log["t"]
    state = log["state"]
    px, py, pz = state[:, 0], state[:, 1], state[:, 2]
    phi, theta, psi = state[:, 6], state[:, 7], state[:, 8]
    pos_des = log["pos_des"]

    frame_idx = np.arange(0, len(t), frame_stride)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    pad = 0.3
    ax.set_xlim(min(px.min(), pos_des[:, 0].min()) - pad, max(px.max(), pos_des[:, 0].max()) + pad)
    ax.set_ylim(min(py.min(), pos_des[:, 1].min()) - pad, max(py.max(), pos_des[:, 1].max()) + pad)
    ax.set_zlim(0, max(pz.max(), pos_des[:, 2].max()) + pad)
    ax.set_xlabel("px [m]"); ax.set_ylabel("py [m]"); ax.set_zlabel("pz [m]")
    ax.set_title(f"{name}: closed-loop trajectory tracking")

    ax.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], "g--", linewidth=1, alpha=0.5, label="reference")
    ax.legend(loc="upper left", fontsize=8)

    trail_line, = ax.plot([], [], [], "b-", linewidth=1, alpha=0.7)
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
        time_text.set_text(f"t = {t[i]:.1f}s")
        return trail_line, arm1, arm2, time_text

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=40, blit=False)
    if shutil.which("ffmpeg"):
        out = PLOTS_DIR / f"{name}_animation.mp4"
        anim.save(out, writer=FFMpegWriter(fps=25))
    else:
        out = PLOTS_DIR / f"{name}_animation.gif"
        anim.save(out, writer=PillowWriter(fps=25))
    print(f"  saved {out}")
    plt.close(fig)


def build_trajectories() -> dict:
    return {
        "straight_line": straight_line_trajectory([0.0, 0.0, 0.0], [2.0, 1.0, 1.0], T=10.0),
        "circle": circle_trajectory(center=(0.0, 0.0, 0.7), radius=1.0, n_laps=1.0, T=20.0),
        "helix": helix_trajectory(center_xy=(0.0, 0.0), radius=0.8, z0=0.3, z1=1.5, n_laps=2.0, T=25.0),
    }


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    trajectories = build_trajectories()
    print(f"{'trajectory':<15} {'RMS err [m]':>12} {'max err [m]':>12} {'final err [m]':>14} {'ITAE':>10}")
    for name, traj in trajectories.items():
        print(f"Simulating '{name}' ({traj.t_end - traj.t_start:.0f}s)...")
        log = simulate(traj)
        m = compute_metrics(log)
        print(f"{name:<15} {m['rms_pos_error_m']:>12.4f} {m['max_pos_error_m']:>12.4f} "
              f"{m['final_pos_error_m']:>14.4f} {m['itae']:>10.4f}")
        make_plots(name, log)
        try:
            make_animation(name, log)
        except ImportError:
            print("  matplotlib animation writer unavailable -- skipping animation")


if __name__ == "__main__":
    main()
