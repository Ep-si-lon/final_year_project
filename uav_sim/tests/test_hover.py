import numpy as np

from uav_sim.src.plant import CrazyflieSim


def test_hover_is_an_equilibrium():
    sim = CrazyflieSim()  # actuator lag starts settled at [m*g, 0, 0, 0]
    hover_cmd = np.array([sim.params.mass * sim.params.g, 0.0, 0.0, 0.0])

    for _ in range(500):  # 1s at dt=0.002
        state = sim.step(hover_cmd, dt=0.002)

    assert np.allclose(state, np.zeros(12), atol=1e-9)
    assert np.allclose(sim.actuator_state, hover_cmd, atol=1e-9)
