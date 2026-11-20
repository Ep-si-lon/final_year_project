"""Physical parameters of the Crazyflie 2.1, per Snider, "CrazyFlie 2.1 Cascaded
PID/PD Control System" (WashU ESE4481, Nov 2025), Eq. 4 and Section 2 text.

Mass and inertia are the report's own values (not, e.g., Bitcraze datasheet or
gym-pybullet-drones estimates used elsewhere) so this model reproduces the
report's own simulated behavior.

Forked verbatim from uav_sim/src/params.py: uav_trajectory_tracking is a
self-contained project (see ASSUMPTIONS_AND_QUESTIONS.md, D1/Q1) and does not
import from uav_sim. If uav_sim's params.py changes, this copy will not pick
it up automatically -- resync by hand if needed.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrazyflieParams:
    mass: float = 0.033  # kg
    g: float = 9.81  # m/s^2

    # Inertia matrix, Eq. 4 [kg*m^2].
    Jxx: float = 16.571710e-6
    Jyy: float = 16.655602e-6
    Jzz: float = 29.261652e-6
    Jxy: float = 0.830806e-6  # reference only; dropped by the gamma reduction below
    Jyz: float = 1.800197e-6  # reference only; dropped by the gamma reduction below
    Jxz: float = 0.718277e-6  # used by the gamma reduction below

    tau_motor: float = 0.1  # s; H(s) = 10/(s+10), Eq. 13

    @property
    def J_full(self) -> np.ndarray:
        """Full symmetric inertia matrix, Eq. 4. Reference only -- the equations
        of motion use only Jxx, Jyy, Jzz, Jxz via the gamma reduction below."""
        return np.array([
            [self.Jxx, self.Jxy, self.Jxz],
            [self.Jxy, self.Jyy, self.Jyz],
            [self.Jxz, self.Jyz, self.Jzz],
        ])


@dataclass(frozen=True)
class Gammas:
    g1: float
    g2: float
    g3: float
    g4: float
    g5: float
    g6: float
    g7: float
    g8: float


def compute_gammas(p: CrazyflieParams) -> Gammas:
    """Standard aircraft-dynamics inertia reduction (report's own MATLAB
    ``inertia_matrix(J)`` function, p.6). Uses only Jxx, Jyy, Jzz, Jxz -- the
    report itself drops the Jxy/Jyz cross terms at this step.

    NOTE (see CHANGELOG.md): the report's own MATLAB literally computes
    ``gamma = Jx*Jy - Jxz^2`` (p.6), but every other gamma formula (g1, g3,
    g4, g7, g8) that divides by this denominator is only consistent with the
    standard ``Jx*Jz - Jxz^2`` form. Jx*Jy ~= 2.76e-10 vs Jx*Jz ~= 4.85e-10
    (~76% different, since Jzz is nearly double Jxx/Jyy) -- this is treated
    as a typo in the report and NOT reproduced here; see CHANGELOG.md."""
    denom = p.Jxx * p.Jzz - p.Jxz ** 2
    return Gammas(
        g1=p.Jxz * (p.Jxx - p.Jyy + p.Jzz) / denom,
        g2=(p.Jzz * (p.Jzz - p.Jyy) + p.Jxz ** 2) / denom,
        g3=p.Jzz / denom,
        g4=p.Jxz / denom,
        g5=(p.Jzz - p.Jxx) / p.Jyy,
        g6=p.Jxz / p.Jyy,
        g7=((p.Jxx - p.Jyy) * p.Jxx + p.Jxz ** 2) / denom,
        g8=p.Jxx / denom,
    )
