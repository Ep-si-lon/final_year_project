"""Reference path generators -- this produces 'r' in the diagram.

Each function returns a ref(t) -> (position[3], velocity[3], acceleration[3], yaw)
closure. Velocity/acceleration are the exact time-derivatives of the position,
computed analytically -- these feed the feedforward term in the controller so
it doesn't have to rely on tracking error alone to "discover" where the
reference is heading.
"""

import numpy as np


def hover(z=1.0):
    def ref(t):
        pos = np.array([0.0, 0.0, z])
        vel = np.zeros(3)
        acc = np.zeros(3)
        yaw = 0.0
        return pos, vel, acc, yaw
    return ref


def circle(radius=1.0, omega=0.3, z=1.0):
    """Circular path in the XY plane at fixed altitude z -- matches the
    reference-tracking sketch from the prof (r drawn as a circle, y and e
    marked against it).

    x(t) = radius*cos(omega*t), y(t) = radius*sin(omega*t)
    velocity/acceleration are its first/second time-derivatives.
    """
    def ref(t):
        c, s = np.cos(omega * t), np.sin(omega * t)
        pos = np.array([radius * c, radius * s, z])
        vel = np.array([-radius * omega * s, radius * omega * c, 0.0])
        acc = np.array([-radius * omega ** 2 * c, -radius * omega ** 2 * s, 0.0])
        yaw = 0.0
        return pos, vel, acc, yaw
    return ref
