"""Closed-loop validation (D11 Stage 2): feedback+feedforward controller
driving the self-contained CrazyflieSim, tracking a step (straight line) and
a circle. Not a re-derivation check like test_flatness.py -- this is "does
the whole loop actually work," measuring RMS/max tracking error.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uav_trajectory_tracking.src.controller import (
    TrajectoryTrackingController, gains_pole_placement_outer,
)
from uav_trajectory_tracking.src.params import CrazyflieParams
from uav_trajectory_tracking.src.plant import CrazyflieSim
from uav_trajectory_tracking.src.reference_trajectories import (
    circle_trajectory, straight_line_trajectory,
)

DT = 0.002


def _run_closed_loop(traj, x0, params=None, dt=DT):
    params = params or CrazyflieParams()
    gains = gains_pole_placement_outer(params)
    controller = TrajectoryTrackingController(params=params, gains=gains, dt=dt, integral_clamp=0.5)
    sim = CrazyflieSim(params=params, x0=x0)
    n_steps = int(round((traj.t_end - traj.t_start) / dt))
    log = np.zeros((n_steps + 1, 12))
    pos_des_log = np.zeros((n_steps + 1, 3))
    log[0] = sim.state
    pos_des_log[0] = traj.eval_pos(traj.t_start, max_deriv=0)[0]
    for i in range(n_steps):
        t = traj.t_start + i * dt
        u4 = controller.compute(sim.state, traj, t)
        log[i + 1] = sim.step(u4, dt=dt)
        pos_des_log[i + 1] = traj.eval_pos(t + dt, max_deriv=0)[0]
    return log, pos_des_log


def test_hover_point_feedback_is_quiescent():
    """At exact tracking (state == trajectory, both at rest), the feedback
    terms should contribute ~nothing -- the controller should reduce to pure
    feedforward, matching uav_sim's own verified hover equilibrium."""
    traj = straight_line_trajectory([0.0, 0.0, 0.5], [0.0, 0.0, 0.5], T=5.0)
    params = CrazyflieParams()
    controller = TrajectoryTrackingController(params=params, gains=gains_pole_placement_outer(params), dt=DT)
    state0 = np.zeros(12)
    state0[2] = 0.5
    u4 = controller.compute(state0, traj, 2.5)
    assert np.isclose(u4[0], params.mass * params.g, atol=1e-9)
    assert np.allclose(u4[1:], 0.0, atol=1e-9)


def test_tracks_straight_line_step():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.5, 0.5])
    traj = straight_line_trajectory(p0, p1, T=8.0)
    x0 = np.zeros(12)
    x0[:3] = p0
    log, pos_des = _run_closed_loop(traj, x0)

    final_pos = log[-1, :3]
    final_err = np.linalg.norm(final_pos - p1)
    print(f"straight-line final position error: {final_err:.4f} m")
    assert final_err < 0.05, final_err

    pos_err = log[:, :3] - pos_des
    rms_err = np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1)))
    print(f"straight-line RMS tracking error: {rms_err:.4f} m")
    assert rms_err < 0.15, rms_err


def test_tracks_circle():
    traj = circle_trajectory(center=(0.0, 0.0, 0.5), radius=1.0, n_laps=1.0, T=20.0)
    x0 = np.zeros(12)
    pos_vel0 = traj.eval_pos(0.0, max_deriv=1)
    x0[:3] = pos_vel0[0]
    x0[3:6] = pos_vel0[1]  # circle starts in motion (tangential vel), not at rest -- match it
    log, pos_des = _run_closed_loop(traj, x0)

    pos_err = log[:, :3] - pos_des
    err_norms = np.linalg.norm(pos_err, axis=1)
    rms_err = np.sqrt(np.mean(err_norms ** 2))
    max_err = err_norms.max()
    print(f"circle RMS tracking error: {rms_err:.4f} m, max: {max_err:.4f} m")
    assert rms_err < 0.15, rms_err
    assert max_err < 0.4, max_err
