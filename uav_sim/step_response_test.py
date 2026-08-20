"""Replicates Section 7 of the reference report: a 0.5 m altitude step,
with a small initial-attitude disturbance standing in for imperfect state
estimation, checking how much the horizontal position (px, py) drifts.

Reference report's own finding: PID reduced the drift rate a lot but never
fully eliminated the steady-state horizontal error. We check whether that
holds here too, and whether the margin_safe preset (Section 6 analysis)
drifts less than the aggressive one.
"""

import numpy as np
from pathlib import Path
from dynamics import QuadrotorDynamics, CrazyflieParams
from controller import CascadedPIDController, GAIN_PRESETS
from actuator import MotorLag
import reference as ref_lib


# Same initial-attitude disturbance as the reference report's Listing 4:
# phi0=0.01 rad, theta0=0.01 rad, psi0=0.02 rad (~0.6, 0.6, 1.1 degrees).
INITIAL_DISTURBANCE = np.array([
    0, 0, 0, 0, 0, 0,       # x,y,z, vx,vy,vz
    0.01, 0.01, 0.02,       # phi, theta, psi
    0, 0, 0,                # p, q, r
])


def run_step(gain_preset="aggressive", duration=10.0, dt=0.002, step_z=0.5):
    params = CrazyflieParams()
    dynamics = QuadrotorDynamics(params, INITIAL_DISTURBANCE.copy())
    controller = CascadedPIDController(params, **GAIN_PRESETS[gain_preset])
    actuator = MotorLag(omega=10.0, u0=[params.mass * 9.81, 0.0, 0.0, 0.0])
    ref_fn = ref_lib.hover(z=step_z)

    n = int(duration / dt)
    log = {"t": [], "pos": []}
    t = 0.0
    for _ in range(n):
        r, v_ref, a_ref, yaw_ref = ref_fn(t)
        u_cmd, e = controller.compute(dynamics.state, r, dt, ref_yaw=yaw_ref,
                                       ref_vel=v_ref, ref_acc=a_ref)
        u = actuator.step(u_cmd, dt)
        dynamics.step(u, dt)
        log["t"].append(t)
        log["pos"].append(dynamics.position.copy())
        t += dt

    log["t"] = np.array(log["t"])
    log["pos"] = np.array(log["pos"])
    return log


def plot(logs, filename="simulation-images/step_response.png"):
    import matplotlib.pyplot as plt

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["x (drift)", "y (drift)", "z (altitude)"]  # pos columns are [x, y, z]
    for col, label in enumerate(labels):
        ax = axes[col]
        for name, log in logs.items():
            ax.plot(log["t"], log["pos"][:, col], label=name)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"{label} [m]")
        ax.set_title(f"{label} response")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    logs = {}
    for preset in GAIN_PRESETS:
        log = run_step(preset)
        drift_x = log["pos"][-1, 0]
        drift_y = log["pos"][-1, 1]
        z_final = log["pos"][-1, 2]
        print(f"{preset:14s} final z={z_final:.4f} m, "
              f"final drift x={drift_x*100:.2f} cm, y={drift_y*100:.2f} cm")
        logs[preset] = log

    plot(logs)
