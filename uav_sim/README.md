# uav_sim

A Python plant model of the Crazyflie 2.1, derived from `Snider_Ian_ESE4481_Final.pdf`
(WashU ESE4481, "CrazyFlie 2.1 Cascaded PID/PD Control System", Ian Snider). This is the
physics simulator only -- no controller -- meant to be driven by force/torque commands
`[Z, L, M, N]` so a controller can be built against it separately.

## Report correspondence

| Module | Report section | Notes |
|---|---|---|
| `src/params.py` | Eq. 4 | mass=0.033kg, full inertia matrix (report's own values, not Bitcraze/gym-pybullet-drones estimates) |
| `src/dynamics.py` | Section 2 (p.2-3) | 12-state nonlinear rigid-body EOM; gamma reduction per report's own `inertia_matrix(J)` (p.5) |
| `src/actuator.py` | Section 5.3, Eq. 13 | motor lag H(s)=10/(s+10) |
| `src/plant.py` | -- | RK4 integration of dynamics+actuator together; `CrazyflieSim` gym-like interface |
| `src/mixer.py` | Eq. 20/22 (p.24-25) | reference-only, not wired into `CrazyflieSim` -- see module docstring |

## Known, disclosed deviations from the report

- **State frame for velocities**: `u, v, w` are body-frame velocities, exactly as in the
  report's Section 2 (some prior work in this repo's history mistakenly used world-frame
  velocities instead -- this module does not).
- **Integrator**: the report's own MATLAB script (Listing 4) uses forward Euler at
  `dt=1/500s`. This module uses RK4 at the same default `dt` -- a disclosed upgrade, not a
  silent fix.
- **Mixer**: the report itself is internally inconsistent between its printed Eq. 22 and its
  own firmware C listing (Listing 5, p.26). `mixer.py` uses Eq. 22 as canonical and flags the
  discrepancy; the mixer is unused by `CrazyflieSim` by default since the report's own
  closed-loop simulation (Section 7) also bypasses it.

## Usage

```python
from uav_sim.src.plant import CrazyflieSim
import numpy as np

sim = CrazyflieSim()
hover = np.array([sim.params.mass * sim.params.g, 0.0, 0.0, 0.0])
state = sim.step(hover, dt=0.002)  # state: [px,py,pz,u,v,w,phi,theta,psi,p,q,r]
```

## Tests

```
pip install -r requirements.txt
pytest tests/ -v
```
