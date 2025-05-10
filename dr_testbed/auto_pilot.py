import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import solve_continuous_are
from dr_testbed.dynamic_model import BasicDynamicModel
import cvxpy as cp

class BasePilot(ABC):
    """
    Base class for all pilots. A pilot is responsible for generating the action
    to be taken by the vehicle base on the current vehicle states, traj and/or
    way_point.
    """
    def __init__(self):
        pass

    def step(self, vehicle_states, traj, way_point):
        """
        Base on the input vehicle states, traj and/or way_point, return the
        action to be taken by the vehicle.
        """
        pass


class DWAPilot(BasePilot):
    """
    A pilot that uses the Dynamic Window Approach to generate the action.
    """

    def __init__(self,
                 model,
                 time_step = 1/60,
                 velocity = 0.3,
                 planning_horizon = 30,
                 steering_angle = 15,
                 num_samples = 31,
                 yaw_weight = 0.3
                 ):
        """
        Args:
            time_step (float): time step of the simulation.
            speed (float): speed of the vehicle.
            planning_horizon (int): number of steps to look ahead.
            steering_angle (float): maximum steering angle of the vehicle.
            num_samples (int): number of samples to generate
        """
        
        assert isinstance(model, BasicDynamicModel), \
            "model should be an instance of BasicDynamicModel."
        self.model = model
        self.time_step = time_step
        self.velocity = velocity
        self.planning_horizon = planning_horizon
        self.steering_angle = steering_angle
        self.num_samples = num_samples
        self.yaw_weight = yaw_weight
    
    def step(self, state, traj, way_point, visuliaze = False):
        sample_angles = np.linspace(-self.steering_angle, self.steering_angle, 
                                    self.num_samples)
        velocities = np.full(self.planning_horizon, self.velocity)
        dts = np.full(self.planning_horizon, self.time_step)
        traj_cost = []
        arrive_steps = []
        for angle in sample_angles:
            steering = np.full(self.planning_horizon, np.radians(angle))
            future_traj = self.model.future_traj(state, velocities, steering, dts)
            future_yaws = future_traj[:, 2]
            future_traj = future_traj[:, :2]
            future_dirs = np.zeros_like(future_traj)
            future_dirs[:, 0] = np.cos(future_yaws)
            future_dirs[:, 1] = np.sin(future_yaws)


            # Calculate the distance between the vehicle and the way point
            dists = np.linalg.norm(future_traj - np.array([way_point.pos]), axis=1)

            dir_diffs = np.linalg.norm(future_dirs - np.array([way_point.dir]), axis=1)
            # yaw_diffs = np.abs(future_yaws - way_point.dir)

            # costs = dists + self.yaw_weight * yaw_diffs
            costs = dists + self.yaw_weight * dir_diffs

            arrive_step = np.argmin(costs)
            cost = costs[arrive_step]
            traj_cost.append(cost)
            arrive_steps.append(arrive_step)

        best_angel_idx = np.argmin(traj_cost)
        best_angle = sample_angles[best_angel_idx]
        arrive_step = arrive_steps[best_angel_idx]

        traj = None
        if visuliaze:
            velocities = np.full(arrive_step, self.velocity)
            steering = np.full(arrive_step, np.radians(best_angle))
            dts = np.full(arrive_step, self.time_step)
            traj = self.model.future_traj(state, velocities, steering, dts)
            traj = traj[:, :2]

        print(f"Best angle: {best_angle}")

        return self.velocity, np.radians(best_angle), traj

class DummyPilot(BasePilot):
    """
    A dummy pilot that run a fixed traj.
    """

    def __init__(self,):
        self.actions = np.load('actions.npy')
        self.count = 0

    def step(self, vehicle_states, traj):
        """
        Update the state of the simulation base on the input actions.

        Args:
            vehicle_states (list): list of states for each vehicle. Each state
                is a tuple of (x, y, yaw).
        
        Returns:
            vehicle_actions (list): list of actions for each vehicle. Each 
                action is a tuple of (velocity, steering).
        """
        if self.count >= len(self.actions):
            self.count = 0
        action = self.actions[self.count]
        self.count += 1
        return action


def normalize_angle(angle):
    """Wrap to [-π, π]."""
    return (angle + np.pi) % (2*np.pi) - np.pi

class PIDPilot(BasePilot):
    def __init__(self,
                 kp_steer=1.0, ki_steer=0.0, kd_steer=0.1,
                 kp_speed=1.0, ki_speed=0.0, kd_speed=0.1,
                 desired_speed=0.5,
                 time_step=1/60):
        # PID gains
        self.kp_s, self.ki_s, self.kd_s = kp_steer, ki_steer, kd_steer
        self.kp_v, self.ki_v, self.kd_v = kp_speed, ki_speed, kd_speed
        self.desired_speed = desired_speed
        self.dt = time_step

        # integrators and previous‐error storage
        self.int_s, self.prev_err_s = 0.0, 0.0
        self.int_v, self.prev_err_v = 0.0, 0.0

        # for velocity estimation
        self.prev_pos = None

    def step(self, state, traj, way_point, visualize=False):
        """
        state: tuple of (x, y, yaw)  ←  NO v here
        way_point.pos: (x_ref, y_ref)
        """
        x, y, yaw = state

        # --- estimate v from position difference ---
        if self.prev_pos is None:
            # first call: assume desired_speed
            v = self.desired_speed
        else:
            dx = x - self.prev_pos[0]
            dy = y - self.prev_pos[1]
            v = np.hypot(dx, dy) / self.dt

        # store for next time
        self.prev_pos = (x, y)

        # --- STEERING PID on heading error ---
        dx_ref = way_point.pos[0] - x
        dy_ref = way_point.pos[1] - y
        desired_yaw = np.arctan2(dy_ref, dx_ref)
        err_s = normalize_angle(desired_yaw - yaw)

        self.int_s += err_s * self.dt
        der_s = (err_s - self.prev_err_s) / self.dt
        steer = (self.kp_s * err_s +
                 self.ki_s * self.int_s +
                 self.kd_s * der_s)
        self.prev_err_s = err_s

        # --- SPEED PID on speed error ---
        err_v = self.desired_speed - v
        self.int_v += err_v * self.dt
        der_v = (err_v - self.prev_err_v) / self.dt
        acc = (self.kp_v * err_v +
               self.ki_v * self.int_v +
               self.kd_v * der_v)
        self.prev_err_v = err_v

        # convert to commanded speed
        v_cmd = v + acc * self.dt

        # clip to physical limits
        max_steer = np.radians(15)
        steer = np.clip(steer, -max_steer, max_steer)
        v_cmd = float(np.clip(v_cmd, 0.0, 5.0))

        return v_cmd, steer, None

class LQRPilot(BasePilot):
    def __init__(self,
                 v0=0.3,
                 L=3.0,
                 Q=None,
                 R=None,
                 dt=1/60,
                 steer_limit_deg=15.0):
        """
        Args:
          v0: cruise speed (m/s)
          L:  wheelbase (m)
          Q:  2x2 state weight
          R:  1x1 input weight
          dt: time step (s)
        """
        self.v0 = v0
        self.L = L
        self.dt = dt
        self.steer_limit = np.radians(steer_limit_deg)

        # Lateral model matrices
        A = np.array([[0.0,    v0],
                      [0.0,    0.0]])
        B = np.array([[0.0],
                      [v0/L]])

        # default weights
        Q = Q if Q is not None else np.eye(2)
        R = R if R is not None else np.eye(1)

        # solve CARE and get gain K
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ (B.T @ P)   # shape (1,2)

    def step(self, state, traj, way_point, visualize=False):
        """
        state: (x, y, psi)
        way_point.pos = (x_ref, y_ref)
        Returns (v_cmd, delta_cmd, _)
        """
        x, y, psi = state

        # 1) Compute reference heading
        x_ref, y_ref = way_point.pos
        dx = x_ref - x
        dy = y_ref - y
        psi_ref = np.arctan2(dy, dx)

        # 2) Cross-track error (signed)
        #    e_y = -sin(psi_ref)*(x - x_ref) + cos(psi_ref)*(y - y_ref)
        e_y = -np.sin(psi_ref)*(x - x_ref) + np.cos(psi_ref)*(y - y_ref)

        # 3) Heading error
        e_psi = normalize_angle(psi - psi_ref)

        # 4) LQR control law: delta = -K [e_y; e_psi]
        err = np.array([e_y, e_psi])
        delta = float(-self.K.dot(err))

        # 5) Clip to steering limit
        delta = np.clip(delta, -self.steer_limit, self.steer_limit)

        # 6) Keep constant speed
        v_cmd = self.v0

        return v_cmd, delta, None
    
class MPCPilot(BasePilot):
    """
    MPC for lateral tracking using the 2-state lateral-error model:
        x_{k+1} = Ad x_k + Bd u_k
    State x = [e_y; e_psi], input u = delta.
    """

    def __init__(self,
                 v0=0.3,
                 L=3.0,
                 dt=1/60,
                 horizon=10,
                 Q=None,
                 R=None,
                 steer_limit_deg=15.0):
        """
        Args:
          v0: cruise speed [m/s]
          L:  wheelbase [m]
          dt: discretization time step [s]
          horizon: prediction horizon (integer)
          Q:  2x2 state weight (defaults to I)
          R:  scalar input weight (defaults to 1)
          steer_limit_deg: max |δ| in degrees
        """
        self.v0 = v0
        self.L = L
        self.dt = dt
        self.N = horizon
        self.steer_limit = np.radians(steer_limit_deg)

        # continuous‐time lateral model:
        #    e_y_dot   =  v0 * e_psi
        #    e_psi_dot = (v0/L)*delta
        A_c = np.array([[0.0,    v0],
                        [0.0,    0.0]])
        B_c = np.array([[0.0],
                        [v0 / L]])

        # discretize with Euler (or zero‐order hold):
        self.Ad = np.eye(2) + A_c * dt
        self.Bd = B_c * dt

        # weights
        self.Q = Q if Q is not None else np.eye(2)
        self.R = R if R is not None else np.eye(1)

        # terminal weight = Q by default
        self.Qf = self.Q

    def step(self, state, traj, way_point, visualize=False):
        """
        state: (x, y, psi)
        way_point.pos = (x_ref, y_ref)
        Returns: (v_cmd, delta_cmd, None)
        """
        x, y, psi = state

        # 1) reference heading
        x_ref, y_ref = way_point.pos
        dx = x_ref - x
        dy = y_ref - y
        psi_ref = np.arctan2(dy, dx)

        # 2) cross‐track error
        e_y = -np.sin(psi_ref)*(x - x_ref) + np.cos(psi_ref)*(y - y_ref)
        # 3) heading error
        e_psi = normalize_angle(psi - psi_ref)

        # initial state‐deviation vector
        x0 = np.array([e_y, e_psi])

        # --- build and solve QP ---
        # variables: x[0..N], u[0..N-1]
        x_var = cp.Variable((2, self.N+1))
        u_var = cp.Variable((1, self.N))

        cost = 0
        constraints = []
        # initial condition
        constraints += [x_var[:,0] == x0]

        # stage cost and dynamics
        for k in range(self.N):
            cost += cp.quad_form(x_var[:,k], self.Q) \
                  + cp.quad_form(u_var[:,k], self.R)
            # dynamics
            constraints += [
                x_var[:,k+1] == self.Ad @ x_var[:,k] + self.Bd.flatten()*u_var[:,k]
            ]
            # steering limits
            constraints += [
                u_var[:,k] <=  self.steer_limit,
               -u_var[:,k] <=  self.steer_limit
            ]

        # terminal cost
        cost += cp.quad_form(x_var[:,self.N], self.Qf)

        # solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        # first control
        delta = float(u_var.value[0,0])
        # clip just in case
        delta = np.clip(delta, -self.steer_limit, self.steer_limit)

        # constant speed
        v_cmd = self.v0

        return v_cmd, delta, None


def saturate(u):
    """Boundary-layer sat: clamp u to [-1,1]."""
    return np.clip(u, -1.0, 1.0)

class RobustPilot(BasePilot):
    """
    Sliding-mode robust steering controller for lateral tracking.
    Keeps a constant cruise speed v0.
    """
    def __init__(self,
                 v0=0.3,
                 L=3.0,
                 lam=0.5,
                 K=1.0,
                 eta=0.2,
                 phi=0.1,
                 steer_limit_deg=15.0):
        """
        Args:
          v0: cruise speed [m/s]
          L:  wheelbase [m]
          lam: sliding-surface weight (e_psi + lam*e_y)
          K:   equiv-control gain on s
          eta: switching gain (robust term)
          phi: boundary layer thickness
          steer_limit_deg: max |δ| [deg]
        """
        self.v0 = v0
        self.L = L
        self.lam = lam
        self.K = K
        self.eta = eta
        self.phi = phi
        self.steer_limit = np.radians(steer_limit_deg)

    def step(self, state, traj, way_point, visualize=False):
        """
        state: (x, y, psi)
        way_point.pos = (x_ref, y_ref)
        Returns (v_cmd, delta_cmd, None)
        """
        x, y, psi = state

        # 1) Reference heading
        x_ref, y_ref = way_point.pos
        dx = x_ref - x
        dy = y_ref - y
        psi_ref = np.arctan2(dy, dx)

        # 2) Lateral error (signed)
        e_y = -np.sin(psi_ref)*(x - x_ref) + np.cos(psi_ref)*(y - y_ref)

        # 3) Heading error
        e_psi = normalize_angle(psi - psi_ref)

        # 4) Sliding surface
        s = e_psi + self.lam * e_y

        # 5) Equivalent control (forces ṡ≈0)
        #    ṡ = ė_ψ + lam ė_y ≈ (v0/L) δ + lam*(v0 e_ψ)  ⇒  δ_eq
        delta_eq = - self.lam * self.L * e_psi \
                   - (self.L * self.K / self.v0) * s

        # 6) Switching control with boundary layer
        delta_sw = - self.eta * saturate(s / self.phi)

        # 7) Combine and clamp
        delta = delta_eq + delta_sw
        delta = float(np.clip(delta, -self.steer_limit, self.steer_limit))

        # 8) Keep constant speed
        v_cmd = self.v0

        return v_cmd, delta, None