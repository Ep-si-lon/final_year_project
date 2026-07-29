"""Wires the block diagram together:
r -> e -> Controller -> u_cmd -> MotorLag -> u_actual -> System -> y,
with y fed back to compute the next e. This loop *is* the "simulation".
"""

import numpy as np
from dynamics import QuadrotorDynamics, CrazyflieParams
from controller import CascadedPIDController, GAIN_PRESETS
from actuator import MotorLag
import reference as ref_lib


def run(duration=20.0, dt=0.005, trajectory="circle", state0=None, motor_bandwidth=10.0,
        gain_preset="aggressive"):
    params = CrazyflieParams()
    if state0 is None:
        state0 = np.zeros(12)  # start at rest on the ground, level attitude

    dynamics = QuadrotorDynamics(params, state0)
    controller = CascadedPIDController(params, **GAIN_PRESETS[gain_preset])
    actuator = MotorLag(omega=motor_bandwidth, u0=[params.mass * 9.81, 0.0, 0.0, 0.0])

    if trajectory == "circle":
        ref_fn = ref_lib.circle(radius=1.0, omega=0.3, z=1.0)
    else:
        ref_fn = ref_lib.hover(z=1.0)

    n_steps = int(duration / dt)
    log = {"t": [], "pos": [], "ref": [], "err": []}

    t = 0.0
    for _ in range(n_steps):
        r, v_ref, a_ref, yaw_ref = ref_fn(t)
        u_cmd, e = controller.compute(dynamics.state, r, dt, ref_yaw=yaw_ref,
                                       ref_vel=v_ref, ref_acc=a_ref)
        u_actual = actuator.step(u_cmd, dt)
        dynamics.step(u_actual, dt)

        log["t"].append(t)
        log["pos"].append(dynamics.position.copy())
        log["ref"].append(r.copy())
        log["err"].append(e.copy())
        t += dt

    for k in ("pos", "ref", "err"):
        log[k] = np.array(log[k])
    log["t"] = np.array(log["t"])
    return log


if __name__ == "__main__":
    import plotting
    log = run()
    print(f"final position error norm: {np.linalg.norm(log['err'][-1]):.4f} m")
    plotting.plot_results(log)
