"""Motor/actuator dynamics -- sits between the controller's commanded
[Z, L, M, N] and the plant's actual input.

A real motor+propeller can't jump to a new thrust instantly: there's
electrical (ESC/driver) and mechanical (rotor inertia, aero spin-up) lag.
The WashU ESE4481 CrazyFlie 2.1 report identifies this as a first-order lag,

    H(s) = 10 / (s + 10)   (bandwidth ~10 rad/s, time constant 0.1 s)

and applies it directly on the aggregate [Z, L, M, N] signal in their
per-axis simulation (rather than per individual motor) -- we do the same
here for consistency with that reference.
"""

import numpy as np


class MotorLag:
    """First-order lag applied independently to each of the 4 channels.

    Exact discretization of du/dt = omega*(u_cmd - u) for a zero-order-hold
    input over one step: u[n+1] = u_cmd + (u[n] - u_cmd) * exp(-omega*dt).
    This is exact for any dt (no integration error), unlike Euler/RK4.
    """

    def __init__(self, omega=10.0, n_channels=4, u0=None):
        self.omega = omega
        self.u = np.zeros(n_channels) if u0 is None else np.array(u0, dtype=float)

    def step(self, u_cmd, dt):
        alpha = np.exp(-self.omega * dt)
        self.u = u_cmd + (self.u - u_cmd) * alpha
        return self.u
