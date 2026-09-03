"""Fuzz testing: random-but-physically-reasonable control inputs, held
constant for a 1s episode, should never make CrazyflieSim silently produce
NaN/Inf or an unbounded blow-up -- EXCEPT via the documented theta=+-90deg
kinematic singularity (see uav_sim/tests/test_singularity.py), which is
detected explicitly here and excluded from the pass/fail assertions rather
than silently ignored.
"""
import numpy as np

from uav_sim.src.plant import CrazyflieSim

N_EPISODES = 100
N_STEPS = 500  # 1.0s at dt=0.002
DT = 0.002

# Local bounds for [Z, L, M, N] -- CrazyflieParams exposes no configurable
# thrust/torque limits, so we define our own "physically reasonable" bounds
# here rather than in src/.
#
# Z: total thrust, sampled in [0, 2 * m * g] -- thrust-to-weight ratio up to
#    2 is a typical quadrotor design target; 0 models a motor-cutout.
# L, M: roll/pitch moments in [-5e-4, 5e-4] N*m.
# N: yaw moment in [-3e-4, 3e-4] N*m.
L_BOUND = 5e-4
M_BOUND = 5e-4
N_BOUND = 3e-4

THETA_IDX = 7
VEL_IDXS = (3, 4, 5)  # u, v, w (body velocities, m/s)
RATE_IDXS = (9, 10, 11)  # p, q, r (body rates, rad/s)
VEL_BOUND = 100.0
RATE_BOUND = 500.0
SINGULARITY_COS_THRESH = 0.05  # |cos(theta)| < this => within ~3deg of +-90deg


def test_random_control_inputs_stay_well_behaved():
    rng = np.random.default_rng(42)

    sim0 = CrazyflieSim()
    z_max = 2.0 * sim0.params.mass * sim0.params.g

    n_hit_singularity = 0
    n_completed_clean = 0

    for _ in range(N_EPISODES):
        Z = rng.uniform(0.0, z_max)
        L = rng.uniform(-L_BOUND, L_BOUND)
        M = rng.uniform(-M_BOUND, M_BOUND)
        N = rng.uniform(-N_BOUND, N_BOUND)
        u_cmd = np.array([Z, L, M, N])

        sim = CrazyflieSim()  # fresh instance, default zero initial state
        hit_singularity = False

        for _ in range(N_STEPS):
            state = sim.step(u_cmd, dt=DT)

            if abs(np.cos(state[THETA_IDX])) < SINGULARITY_COS_THRESH:
                hit_singularity = True
                break

            assert np.all(np.isfinite(state)), (
                f"non-finite state with u_cmd={u_cmd}: {state}"
            )
            for i in VEL_IDXS:
                assert abs(state[i]) < VEL_BOUND, (
                    f"body velocity blew up (idx {i}={state[i]}) with u_cmd={u_cmd}"
                )
            for i in RATE_IDXS:
                assert abs(state[i]) < RATE_BOUND, (
                    f"body rate blew up (idx {i}={state[i]}) with u_cmd={u_cmd}"
                )

        if hit_singularity:
            n_hit_singularity += 1
        else:
            n_completed_clean += 1

    # Informational only -- any split from 0 to 100 is fine, since inputs are
    # random. Run with -s to see this printed.
    print(
        f"\n[fuzz] {n_hit_singularity}/{N_EPISODES} episodes hit the known "
        f"theta=+-90deg singularity; {n_completed_clean}/{N_EPISODES} "
        f"completed cleanly within generous finite bounds."
    )
    assert n_hit_singularity + n_completed_clean == N_EPISODES
