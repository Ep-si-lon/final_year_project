"""3D animated visualization of a logged flight -- pure matplotlib, no
simulation software involved (the motion itself comes entirely from
uav_sim's own hand-written dynamics/controller, this file only draws it).

Produces:
  - flight_animation.mp4  (or .gif if ffmpeg is unavailable): a movie of the
    drone flying the logged trajectory, orientation included.
  - flight_filmstrip.png: a static grid of snapshots, for viewing inline
    wherever an actual video can't be played.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def rotation_matrix(phi, theta, psi):
    """Body -> world rotation, R = Rz(psi) @ Ry(theta) @ Rx(phi) -- the same
    convention used in dynamics.py's acceleration equations."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    Rx = np.array([[1, 0, 0], [0, cphi, -sphi], [0, sphi, cphi]])
    Ry = np.array([[ctheta, 0, stheta], [0, 1, 0], [-stheta, 0, ctheta]])
    Rz = np.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def drone_body_points(pos, rpy, arm_len=0.18):
    """World-frame endpoints of the X-configuration arms and a short nose
    marker indicating heading (body +x). Sized for visibility, not to scale
    (the real Crazyflie 2.1 is ~9.2cm across)."""
    phi, theta, psi = rpy
    R = rotation_matrix(phi, theta, psi)

    body_pts = arm_len * np.array([
        [1, 1, 0], [-1, -1, 0],   # arm 1 (diagonal)
        [1, -1, 0], [-1, 1, 0],   # arm 2 (diagonal)
    ]) / np.sqrt(2)
    nose = arm_len * np.array([1.6, 0, 0])

    world_pts = (R @ body_pts.T).T + pos
    world_nose = (R @ nose) + pos
    return world_pts, world_nose


def _draw_frame(ax, pos, rpy, arm_len, color="black"):
    world_pts, nose = drone_body_points(pos, rpy, arm_len)
    lines = [
        ax.plot(*zip(world_pts[0], world_pts[1]), color=color, lw=2)[0],
        ax.plot(*zip(world_pts[2], world_pts[3]), color=color, lw=2)[0],
        ax.plot(*zip(pos, nose), color="red", lw=2)[0],
        ax.plot(*[[p] for p in pos], "o", color=color, markersize=5)[0],
    ]
    return lines


def animate_flight(log, filename=None, fps=30, playback_speed=4.0, arm_len=0.18):
    t, pos, att, ref = log["t"], log["pos"], log["att"], log["ref"]
    dt = t[1] - t[0]
    frame_stride = max(1, int(round((1.0 / fps) * playback_speed / dt)))
    idx = np.arange(0, len(t), frame_stride)

    has_ffmpeg = shutil.which("ffmpeg") is not None
    if filename is None:
        filename = "flight_animation.mp4" if has_ffmpeg else "flight_animation.gif"

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], "--", color="tab:blue", alpha=0.5, label="reference (r)")
    trail_line, = ax.plot([], [], [], color="tab:orange", lw=1, label="flown path (y)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(loc="upper left")
    pad = 0.3
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    ax.set_zlim(max(0, pos[:, 2].min() - pad), pos[:, 2].max() + pad)

    drone_artists = []

    def update(i):
        nonlocal drone_artists
        for artist in drone_artists:
            artist.remove()
        frame = idx[i]
        drone_artists = _draw_frame(ax, pos[frame], att[frame], arm_len)
        trail_line.set_data(pos[:frame, 0], pos[:frame, 1])
        trail_line.set_3d_properties(pos[:frame, 2])
        ax.set_title(f"t = {t[frame]:.2f} s")
        return drone_artists + [trail_line]

    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    writer = FFMpegWriter(fps=fps) if has_ffmpeg else PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=110)
    plt.close(fig)
    print(f"Saved animation to {filename} ({len(idx)} frames at {fps}fps, "
          f"{playback_speed}x real time)")
    return filename


def filmstrip(log, n_frames=6, filename="flight_filmstrip.png", arm_len=0.18):
    t, pos, att, ref = log["t"], log["pos"], log["att"], log["ref"]
    frame_idx = np.linspace(0, len(t) - 1, n_frames).astype(int)

    fig = plt.figure(figsize=(4 * n_frames, 4.5))
    for k, frame in enumerate(frame_idx):
        ax = fig.add_subplot(1, n_frames, k + 1, projection="3d")
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], "--", color="tab:blue", alpha=0.4)
        ax.plot(pos[:frame + 1, 0], pos[:frame + 1, 1], pos[:frame + 1, 2],
                 color="tab:orange", lw=1)
        _draw_frame(ax, pos[frame], att[frame], arm_len)
        pad = 0.3
        ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
        ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
        ax.set_zlim(max(0, pos[:, 2].min() - pad), pos[:, 2].max() + pad)
        ax.set_title(f"t = {t[frame]:.2f} s")
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)
    print(f"Saved filmstrip to {filename}")
    return filename


if __name__ == "__main__":
    import simulate
    log = simulate.run(duration=20.0, trajectory="circle", gain_preset="aggressive")
    filmstrip(log)
    animate_flight(log)
