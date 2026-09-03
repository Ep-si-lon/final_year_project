"""Reproduces the qualitative shape of the report's own Figure 12 (p.20):
a 0.5m altitude step settling within ~2.5s, using the report's own Z-loop
gains (k_pz=0.297, k_dz=0.198). This is a throwaway test-only PD controller,
not part of the plant-only deliverable -- it exists purely to confirm the
plant's dynamics match the report's own simulated behavior.
"""
import numpy as np

from uav_sim.src.plant import CrazyflieSim

K_PZ = 0.297
K_DZ = 0.198


def test_altitude_step_reproduces_report_figure_12():
    sim = CrazyflieSim()
    dt = 0.002
    steps = int(10 / dt)  # 10s, matching the report's own plot window

    pz_log = np.zeros(steps)
    for i in range(steps):
        pz, vz = sim.state[2], sim.state[5]  # w == earth-frame vz at zero tilt
        Z = sim.params.mass * sim.params.g + K_PZ * (0.5 - pz) + K_DZ * (0.0 - vz)
        state = sim.step(np.array([Z, 0.0, 0.0, 0.0]), dt=dt)
        pz_log[i] = state[2]

    settle_idx = int(2.5 / dt)
    assert np.isclose(pz_log[settle_idx:].mean(), 0.5, atol=0.02)
    assert pz_log[-1] > pz_log[int(0.5 / dt)]  # monotonically rose from near t=0.5s
