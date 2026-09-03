"""Time-varying setpoint generators, returning ref6 = [px_c,py_c,pz_c,
vx_c,vy_c,vz_c] (earth frame) for CascadedController.compute -- includes the
exact tangential velocity, not just zero, so the outer-loop velocity term
isn't fighting a moving target it's never told the speed of.
"""
import numpy as np


def hover(t: float, x: float = 0.0, y: float = 0.0, z: float = 0.5) -> np.ndarray:
    return np.array([x, y, z, 0.0, 0.0, 0.0])


def circle(
    t: float,
    radius: float = 1.0,
    z: float = 10.0,
    omega: float = 0.5,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> np.ndarray:
    """Circle of given radius in the horizontal plane at altitude z, angular
    rate omega [rad/s] (period = 2*pi/omega). omega=0.5 keeps the required
    centripetal tilt small (~1.5deg for radius=1m: tilt ~= omega^2*radius/g),
    staying within the controller's small-angle design assumption."""
    px_c = center_x + radius * np.cos(omega * t)
    py_c = center_y + radius * np.sin(omega * t)
    vx_c = -radius * omega * np.sin(omega * t)
    vy_c = radius * omega * np.cos(omega * t)
    return np.array([px_c, py_c, z, vx_c, vy_c, 0.0])
