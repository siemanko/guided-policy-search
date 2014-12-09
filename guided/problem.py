import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

from .controller import LQR, EnergyStabilization, Controller, MixedController
from .plant import DoublePendulum

class DoublePendulumProblem(object):
    def __init__(self, params=None):
        self.params = {
            'g_ms2': 9.8, # acceleration due to gravity, in m/s^2
            'l1_m': 1.0, # length of pendulum 1 in m
            'l2_m': 2.0, # length of pendulum 2 in m
            'm1_kg': 1.0, # mass of pendulum 1 in kg
            'm2_kg': 1.0, # mass of pendulum 2 in kg
            'dt': 0.01,
            'control_limit': [-40.0, 40.0]
        }
        self.params.update(params or {})
        self._cached_exp_time_len = None
        self._cached_exp_time = None
        self.goal_state = np.array([pi, 0.0, pi, 0.0])

    def plant(self):
        return DoublePendulum(self.params)

    def energy_shaping(self):
        p = self.plant()

        # Energy controller
        energy_ctrl = EnergyStabilization(p, self.goal_state, 0)

        C=1.0
        k = [C*0.1, C*0.025, C*0.025]

        max_lqr_score = 19.0

        # Erection controller
        normalize_angle = lambda angle : angle % (2*pi) - pi

        def angle_controller(state, t):
            t1 = state[0]
            t2 = state[2]
            diff = t1-t2
            if abs(diff) > pi:
                new_abs = 2*pi - abs(diff)
                diff = new_abs * np.sign(diff)
            return diff

        straight_ctrl = Controller(angle_controller)
        straight_dot_ctrl = Controller(lambda state, t: -(state[3]-state[1]) )

        swingup_controller = MixedController()
        swingup_controller.add(k[0], energy_ctrl)
        swingup_controller.add(k[1], straight_ctrl)
        swingup_controller.add(k[2], straight_dot_ctrl)

        # LQR controller focused on upside
        lqr = LQR(p, self.goal_state, 0)
        swingup_controller.add_lqr(lqr, max_lqr_score)

        # Set the mixed controller
        p.set_controller(swingup_controller.get())
        return p

    def compute_trajectory(self, p, initial_state, time_horizon):
        # simulation and visualization
        y = p.simulate(initial_state, time_horizon, visualize=False)
        time_range = np.arange(0.0, time_horizon + p.dt, p.dt)
        u = [p.controller(y[i,:], time_range[i]) for i in range(y.shape[0])]
        u = np.vstack(u)
        return (y, u)

    def sample_trajectory(self, p):
        downside_state = np.array([0.0, 0.0, 0.0, 0.0])
        initial_state = downside_state + np.array([1.0, 0.0, 1.0, 0.0]) * np.random.normal(0.0, 0.1)
        return self.compute_trajectory(p, initial_state)

    def exp_time(self, how_many):
        COEFF = 2
        if (self._cached_exp_time_len is None or
                self._cached_exp_time_len != how_many):
            self._cached_exp_time_len = how_many
            self._cached_exp_time = \
                np.array([np.exp(COEFF*float(how_many-i)/float(how_many))
                          for i in range(how_many)])
        return self._cached_exp_time

    def cost(self, trajectory):
        # the higher the more cost matters with time.
        x, u = trajectory
        position_diff = np.sqrt( (self.goal_state[0] - x[:, 0])**2 +
                                 (self.goal_state[2] - x[:, 2])**2)
        u_lb, u_ub = self.params['control_limit']
        torque = np.linalg.norm(np.maximum(np.minimum(u, u_ub), u_lb))
        total_cost = (position_diff + torque) * self.exp_time(x.shape[0])
        return sum(total_cost)

    def score_distribution(self, p, trajectories):
        """To get uncertainty of the mean divide by sqrt(len(trajctories))"""
        avg_cost = 0.0
        c = []
        for traj in trajectories:
            c.append(self.cost(traj))
        c = np.array(c)
        avg = sum(c) / c.shape[0]
        std = (c - avg)*(c - avg)
        std = np.sqrt(sum(std)/(c.shape[0]-1))
        return avg, std

    def animate_trajectory(self, plant, trajectory):
        x, _ = trajectory
        return plant.animation(x)

    def ml_data_from_trajectories(self, trajectories):
        X, Y, weights = [], [], []

        for trajectory in trajectories:
            x, u = trajectory
            c =  self.cost(trajectory)
            for i in range(x.shape[0]):
                X.append(x[i, :])
                Y.append(u[i])
                weights.append(1.0/c)

        return (np.vstack(X), np.hstack(Y), np.hstack(weights))

    def plot_u(self, p, trajectory):
        x, u = trajectory
        u_lb, u_ub = self.params['control_limit']
        torque = np.maximum(np.minimum(u, u_ub), u_lb)
        plt.figure()
        time_range = np.arange(0, x.shape[0]*p.dt, p.dt)
        plt.scatter(time_range, torque)

    def features(self, x):
        result = np.zeros((x.shape[0], 7))
        t1,v1,t2,v2 = x[:,0], x[:,1], x[:,2], x[:,3]
        result[:,0:4] = x
        result[:,4] = np.cos(t1)
        result[:,5] = np.cos(t2)
        result[:,6] = v1*v2
        return result
