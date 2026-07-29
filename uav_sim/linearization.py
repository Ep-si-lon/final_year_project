"""Linearize the nonlinear dynamics at hover and check controllability.

Mirrors Sections 2-3 of the WashU ESE4481 report: find A = df/dx, B = df/du
at the hover equilibrium, then check rank([B AB A^2B ... A^11B]) to see how
much of the 12-state system is actually reachable through the 4 inputs
[F, tau_phi, tau_theta, tau_psi]. The report finds rank 8/12 -- horizontal
position/velocity are only reachable indirectly, through attitude.
"""

import numpy as np
from dynamics import QuadrotorDynamics, CrazyflieParams, G

STATE_NAMES = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]


def linearize(params=None, eps=1e-6):
    """Central-difference Jacobian of the nonlinear dynamics at hover
    (all states zero, thrust = weight, zero torques)."""
    params = params if params is not None else CrazyflieParams()
    plant = QuadrotorDynamics(params)

    x0 = np.zeros(12)
    u0 = np.array([params.mass * G, 0.0, 0.0, 0.0])

    n, m = 12, 4
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        f_plus = plant._deriv(x0 + dx, u0)
        f_minus = plant._deriv(x0 - dx, u0)
        A[:, i] = (f_plus - f_minus) / (2 * eps)

    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        f_plus = plant._deriv(x0, u0 + du)
        f_minus = plant._deriv(x0, u0 - du)
        B[:, j] = (f_plus - f_minus) / (2 * eps)

    return A, B


def controllability_matrix(A, B):
    n = A.shape[0]
    blocks = [B]
    Ak_B = B
    for _ in range(1, n):
        Ak_B = A @ Ak_B
        blocks.append(Ak_B)
    return np.hstack(blocks)


def controllable_subspace(Wc, tol=1e-9):
    """Which individual states have a nonzero row in Wc (necessary, not
    sufficient, condition for that state being reachable) -- a quick,
    interpretable diagnostic alongside the rank number."""
    row_norms = np.linalg.norm(Wc, axis=1)
    return row_norms > tol


if __name__ == "__main__":
    A, B = linearize()
    Wc = controllability_matrix(A, B)
    rank = np.linalg.matrix_rank(Wc)

    print(f"A: {A.shape}, B: {B.shape}, controllability matrix Wc: {Wc.shape}")
    print(f"rank(Wc) = {rank} / 12")

    reachable = controllable_subspace(Wc)
    print("\nPer-state row activity in Wc (nonzero => that state appears in the reachable chain):")
    for name, active in zip(STATE_NAMES, reachable):
        print(f"  {name:6s}: {'yes' if active else 'no'}")

    if rank < 12:
        print(f"\nSystem is NOT fully controllable in all 12 states ({rank}/12), "
              "matching the reference report's finding. Horizontal position/velocity "
              "(x, y, vx, vy) are only reachable indirectly, through commanded tilt "
              "(phi, theta) -- which is exactly why the controller cascades position "
              "-> attitude rather than commanding x/y directly.")
    else:
        print(f"\nSystem IS fully controllable ({rank}/12) -- this differs from the "
              "reference report's rank-8 finding. The reason: A[vx,theta]=+g and "
              "A[vy,phi]=-g here (the gravity/thrust projection cos(phi)*sin(theta) "
              "etc. is included directly in the world-frame acceleration equations), "
              "so tilting the drone couples straight into horizontal acceleration in "
              "the linearization -- the standard result (e.g. Beard & McLain). The "
              "reference report's body-frame u_dot/v_dot equations only place gravity "
              "on the body-z axis ((0,0,Z-mg)/m) with no sin(theta)/sin(phi) terms in "
              "the u_dot/v_dot rows, which structurally zeroes those two couplings and "
              "caps their rank at 8. Position/velocity are still only reachable "
              "*indirectly* here too (no direct x/y actuation, hence still cascading "
              "position -> attitude) -- full rank means the coupling chain reaches "
              "every state eventually, not that it's direct.")
