"""First-order motor/actuator lag, per Snider Section 5.3, Eq. 13:
H(s) = 10/(s+10), i.e. tau = 0.1s, applied identically to each of [Z, L, M, N]
between the commanded and actual value.

The report's own discrete H(z) form (Eq. 12) bakes in a PWM-normalization
scale factor and is not a drop-in equivalent of this continuous form -- not
needed for a plant that takes force/torque commands directly (see mixer.py).

Forked verbatim from uav_sim/src/actuator.py -- see params.py's docstring.
"""
import numpy as np


def lag_deriv(lag_state4: np.ndarray, u4_cmd: np.ndarray, tau: float) -> np.ndarray:
    """d(actual)/dt for the first-order lag: actual -> cmd with time constant tau."""
    return (u4_cmd - lag_state4) / tau
