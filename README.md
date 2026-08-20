# Crazyflie 2.1 circular-trajectory control

This repository simulates a nonlinear 12-state Crazyflie 2.1 model with a
cascaded controller: outer X/Y PID, inner roll/pitch PD, altitude PD and yaw
PD.  The drone takes off to 1 m and tracks a 1 m-radius circular reference.

## PSO tuning

`uav_sim/pso_tune.py` uses deterministic particle swarm optimization to tune
the shared outer-loop `Kp`, `Ki` and `Kd`.  All other gains and the plant are
held fixed.  A small constant horizontal acceleration is applied only during
tuning and disturbance validation so the integral gain is selected against a
real steady-state-error case; normal simulations are unchanged.

From `uav_sim/`, run:

```bash
python pso_tune.py
python -c "from simulate import run; print(run(duration=60, gain_preset='pso_optimized')['err'][-1])"
```

The seeded run writes the exact gains, metrics and convergence history to
`uav_sim/pso_results/pso_results.json` and the before/after plot to
`uav_sim/simulation-images/pso_comparison.png`. All generated PNG plots and
filmstrips are kept in `uav_sim/simulation-images/`; MP4 flight animations are
kept in `uav_sim/simulation/`.
