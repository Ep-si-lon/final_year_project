"""Nonlinear rigid-body dynamics model of the Crazyflie 2.1.

This is the 'System / UAV' block in the r -> e -> Controller -> u -> System -> y
diagram: it takes a control input u and integrates the equations of motion
forward in time to produce the next state (from which y is read).
"""

import numpy as np

G = 9.81  # m/s^2


class CrazyflieParams:
    """Physical parameters for the Crazyflie 2.1.

    mass: official Bitcraze datasheet ("Datasheet Crazyflie 2.1 - Rev 3"),
    section 8, takeoff weight = 29 g.
    https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf

    Ixx/Iyy/Izz/max_thrust: Bitcraze does not publish an inertia matrix or a
    thrust rating in the datasheet (getting these requires a system-ID
    measurement, not a spec-sheet number), so these are the values used by
    the gym-pybullet-drones cf2x model (Panerati et al., "Learning to Fly",
    IROS 2021 / arXiv:2103.02142), itself built on Crazyflie system
    identification work. Verify against your own measurements if precision
    matters for your results.
    """

    mass = 0.029        # kg  -- Bitcraze datasheet takeoff weight
    Ixx = 1.4e-5        # kg*m^2  -- gym-pybullet-drones cf2x
    Iyy = 1.4e-5        # kg*m^2  -- gym-pybullet-drones cf2x
    Izz = 2.17e-5       # kg*m^2  -- gym-pybullet-drones cf2x
    max_thrust = 0.64   # N, total over 4 motors (thrust-to-weight 2.25, gym-pybullet-drones cf2x)
    max_torque = 3e-4   # N*m, per axis


class QuadrotorDynamics:
    """Euler-angle 6-DOF rigid-body model.

    State x = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]  (12,)
        position, velocity (world frame), roll/pitch/yaw, body angular rates
    Input u = [F, tau_phi, tau_theta, tau_psi]
        total thrust + body-frame torques (the 'virtual' control input;
        a motor mixer can map this to 4 individual rotor speeds later)
    """

    def __init__(self, params=None, state0=None):
        self.p = params if params is not None else CrazyflieParams()
        self.state = np.zeros(12) if state0 is None else np.array(state0, dtype=float)

    def _deriv(self, x, u):
        _, _, _, vx, vy, vz, phi, theta, psi, p, q, r = x
        F, tau_phi, tau_theta, tau_psi = u

        F = np.clip(F, 0.0, self.p.max_thrust)
        tau_phi = np.clip(tau_phi, -self.p.max_torque, self.p.max_torque)
        tau_theta = np.clip(tau_theta, -self.p.max_torque, self.p.max_torque)
        tau_psi = np.clip(tau_psi, -self.p.max_torque, self.p.max_torque)

        m = self.p.mass
        ax = (F / m) * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi))
        ay = (F / m) * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi))
        az = (F / m) * (np.cos(phi) * np.cos(theta)) - G

        phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

        p_dot = (tau_phi + (self.p.Iyy - self.p.Izz) * q * r) / self.p.Ixx
        q_dot = (tau_theta + (self.p.Izz - self.p.Ixx) * p * r) / self.p.Iyy
        r_dot = (tau_psi + (self.p.Ixx - self.p.Iyy) * p * q) / self.p.Izz

        return np.array([vx, vy, vz, ax, ay, az, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])

    def step(self, u, dt):
        """Advance the state by dt (RK4) given control input u. This is the
        numerical-integration step that stands in for a physics engine."""
        x = self.state
        k1 = self._deriv(x, u)
        k2 = self._deriv(x + 0.5 * dt * k1, u)
        k3 = self._deriv(x + 0.5 * dt * k2, u)
        k4 = self._deriv(x + dt * k3, u)
        self.state = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.state

    @property
    def position(self):
        return self.state[0:3]

    @property
    def velocity(self):
        return self.state[3:6]

    @property
    def attitude(self):
        return self.state[6:9]  # phi, theta, psi
