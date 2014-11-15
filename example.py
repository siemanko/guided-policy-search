import numpy as np

from numpy import pi

from controller import LQR
from plant import DoublePendulum

DOUBLE_PENDULUM_PARAMS = {
    'g_ms2': 9.8, # acceleration due to gravity, in m/s^2
    'l1_m': 1.0, # length of pendulum 1 in m
    'l2_m': 1.0, # length of pendulum 2 in m
    'm1_kg': 1.0, # mass of pendulum 1 in kg
    'm2_kg': 1.0, # mass of pendulum 2 in kg
    'control_limit': [-15.0, 15.0]
}


def DP_lqr():
    """Double Pendulum upright LQR stabilization example"""
    p = DoublePendulum(DOUBLE_PENDULUM_PARAMS)

    upside_state = np.array([pi, 0.0, pi, 0.0])
    # upside_state with noise in position
    perturbation = np.array([1.0, 0.0, 1.0, 0.0]) * np.random.normal(0.0, 0.05, (4,))
    perturbation = np.array([0.06, 0.0, -0.06, 0.0])
    initial_state = upside_state + perturbation
    print 'Starting perturbed by %s' % (perturbation,)
    # 20 seconds
    time_range = np.arange(0.0, 20, 0.01)

    # LQR controller focused on upside
    lqr = LQR().train(p, upside_state, 0)
    p.set_controller(lqr.get())

    # simulation and visualization
    y = p.simulate(initial_state, time_range, visualize=True)



def swingup_example():
    """Double Pendulum upright LQR stabilization example"""
    p = DoublePendulum(DOUBLE_PENDULUM_PARAMS)

    upside_state = np.array([0.0, 0.0, 0.0, 0.0])
    # upside_state with noise in position
    initial_state = upside_state + np.array([1.0, 0.0, 1.0, 0.0]) * np.random.normal(0.0, 0.4)
    print 'Starting from %s' % (initial_state,)

    # 20 seconds
    time_range = np.arange(0.0, 20, 0.01)

    # LQR controller focused on upside
    lqr = LQR().train(p, upside_state, 0)

    # Energy controller
    energy_ctrl = EnergyStabilization(upside_state, 0.01)


    p.set_controller(energy_ctrl.get())

    # simulation and visualization
    y = p.simulate(initial_state, time_range, visualize=True)


if __name__ == '__main__':
    DP_lqr()
    #swingup_example()
