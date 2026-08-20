"""Independent physics cross-check: our hand-written dynamics.py vs. the
PyBullet-based gym-pybullet-drones engine (a mature, widely-used, third-party
rigid-body simulator).

Per project scope, this is a VALIDATION TOOL ONLY -- uav_sim/dynamics.py
remains the actual deliverable (per-instructions, pure Python math modeling).
This script asks: does an independent physics engine agree with our
equations, especially on the point that was in question earlier (does
tilting the drone actually couple into horizontal acceleration, i.e. is our
world-frame model -- which found full rank-12 controllability -- physically
correct, versus the reference PDF's simplified body-frame model which found
rank 8)?

Requires: pip install -e <clone of utiasDSL/gym-pybullet-drones> (with
gym_pybullet_drones/assets/cf2x.urdf's mass corrected from 0.027 to 0.029 kg
to match the Bitcraze datasheet value used in dynamics.py.CrazyflieParams).
"""

import numpy as np
from pathlib import Path
from dynamics import QuadrotorDynamics, CrazyflieParams, G

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def test_vertical_thrust(duration=1.0):
    """Open-loop: command 10% above hover thrust on all 4 motors, compare
    the resulting vertical acceleration/position between engines."""
    params = CrazyflieParams()

    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                      initial_xyzs=np.array([[0, 0, 1.0]]),
                      initial_rpys=np.array([[0, 0, 0]]),
                      physics=Physics.PYB, pyb_freq=240, ctrl_freq=240, gui=False)
    dt = 1.0 / 240
    n_steps = int(duration / dt)

    F_cmd = 1.1 * params.mass * G
    rpm = np.sqrt(F_cmd / (4 * env.KF)) * np.ones(4)

    pb_z = []
    obs, _ = env.reset()
    for _ in range(n_steps):
        obs, _, _, _, _ = env.step(rpm.reshape(1, 4))
        pb_z.append(obs[0, 2])
    env.close()
    pb_z = np.array(pb_z)

    plant = QuadrotorDynamics(params, np.array([0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    ours_z = []
    for _ in range(n_steps):
        plant.step(np.array([F_cmd, 0, 0, 0]), dt)
        ours_z.append(plant.position[2])
    ours_z = np.array(ours_z)

    t = np.arange(n_steps) * dt
    print("=== Test 1: vertical thrust (F = 1.1x hover weight) ===")
    print(f"final z:  pybullet={pb_z[-1]:.4f} m   ours={ours_z[-1]:.4f} m   "
          f"diff={abs(pb_z[-1]-ours_z[-1])*1000:.2f} mm")
    print(f"max |diff| over trajectory: {np.abs(pb_z-ours_z).max()*1000:.2f} mm")
    return t, pb_z, ours_z


def test_roll_coupling(duration=0.3, delta_frac=0.002):
    """Open-loop: near-hover thrust + a small constant differential RPM to
    induce roll, on BOTH engines with the same commanded [F, tau_x, 0, 0].
    Checks whether attitude (phi) couples into lateral acceleration/velocity
    the way dynamics.py assumes -- and whether it does so consistently in an
    independent physics engine."""
    params = CrazyflieParams()

    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                      initial_xyzs=np.array([[0, 0, 1.0]]),
                      initial_rpys=np.array([[0, 0, 0]]),
                      physics=Physics.PYB, pyb_freq=240, ctrl_freq=240, gui=False)
    dt = 1.0 / 240
    n_steps = int(duration / dt)

    hover_rpm = env.HOVER_RPM
    delta = delta_frac * hover_rpm
    # prop layout (cf2x): 0,1 at y=-0.028 ; 2,3 at y=+0.028 -- differential
    # here creates a roll moment about the x-axis.
    rpm = np.array([hover_rpm - delta, hover_rpm - delta, hover_rpm + delta, hover_rpm + delta])
    tau_x_equiv = env.L * env.KF * ((rpm[2] ** 2 + rpm[3] ** 2) - (rpm[0] ** 2 + rpm[1] ** 2)) / np.sqrt(2)

    pb_phi, pb_vy, pb_y = [], [], []
    obs, _ = env.reset()
    for _ in range(n_steps):
        obs, _, _, _, _ = env.step(rpm.reshape(1, 4))
        pb_phi.append(obs[0, 7])   # roll
        pb_vy.append(obs[0, 11])   # world-frame vy
        pb_y.append(obs[0, 1])
    env.close()
    pb_phi, pb_vy, pb_y = np.array(pb_phi), np.array(pb_vy), np.array(pb_y)

    plant = QuadrotorDynamics(params, np.array([0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    ours_phi, ours_vy, ours_y = [], [], []
    F_cmd = params.mass * G
    for _ in range(n_steps):
        plant.step(np.array([F_cmd, tau_x_equiv, 0, 0]), dt)
        ours_phi.append(plant.attitude[0])
        ours_vy.append(plant.velocity[1])
        ours_y.append(plant.position[1])
    ours_phi, ours_vy, ours_y = np.array(ours_phi), np.array(ours_vy), np.array(ours_y)

    t = np.arange(n_steps) * dt

    # Self-consistency check using ONLY pybullet's own data: does its lateral
    # velocity match integrating -g*sin(phi) over its own recorded phi(t)?
    predicted_vy_from_pb_phi = np.concatenate([[0], np.cumsum(-G * np.sin(pb_phi[:-1]) * dt)])
    corr = np.corrcoef(pb_vy, predicted_vy_from_pb_phi)[0, 1]

    print("\n=== Test 2: roll torque -> lateral (y) coupling ===")
    print(f"commanded equivalent tau_x = {tau_x_equiv:.2e} N*m")
    print(f"final phi:  pybullet={np.degrees(pb_phi[-1]):.3f} deg   ours={np.degrees(ours_phi[-1]):.3f} deg")
    print(f"final vy:   pybullet={pb_vy[-1]:.4f} m/s   ours={ours_vy[-1]:.4f} m/s")
    print(f"final y:    pybullet={pb_y[-1]:.4f} m   ours={ours_y[-1]:.4f} m")
    print(f"self-consistency (pybullet vy vs integral of -g*sin(phi) from pybullet's OWN phi log): "
          f"correlation={corr:.4f}")
    return t, (pb_phi, pb_vy, pb_y), (ours_phi, ours_vy, ours_y)


def plot(t1, pb_z, ours_z, t2, pb_data, ours_data,
         filename="simulation-images/pybullet_validation.png"):
    import matplotlib.pyplot as plt
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pb_phi, pb_vy, pb_y = pb_data
    ours_phi, ours_vy, ours_y = ours_data

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(t1, pb_z, label="pybullet")
    axes[0].plot(t1, ours_z, "--", label="ours")
    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("z [m]")
    axes[0].set_title("Test 1: vertical thrust")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(t2, np.degrees(pb_phi), label="pybullet")
    axes[1].plot(t2, np.degrees(ours_phi), "--", label="ours")
    axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("phi [deg]")
    axes[1].set_title("Test 2: roll angle")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(t2, pb_y, label="pybullet")
    axes[2].plot(t2, ours_y, "--", label="ours")
    axes[2].set_xlabel("time [s]"); axes[2].set_ylabel("y [m]")
    axes[2].set_title("Test 2: lateral drift from roll")
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    t1, pb_z, ours_z = test_vertical_thrust()
    t2, pb_data, ours_data = test_roll_coupling()
    plot(t1, pb_z, ours_z, t2, pb_data, ours_data)
