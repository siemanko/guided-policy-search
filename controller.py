import control
import numpy as np

class Controller(object):
    pass

class LQR(Controller):
    def __init__(self, plant, state0, t0, Q=None, R=None):
        A = plant.df(state0, t0)
        B = plant.control_matrix(state0, t0)
        Q = Q or np.identity(A.shape[1])
        R = R or np.identity(B.shape[1]) * 0.01

        K, S, E = control.lqr(A, B, Q, R)
        self.state0 = state0.copy()
        self.K = K
        self.S = S
        self.E = E

    def get(self):
        controller =  lambda state, t: np.dot(- self.K, (state - self.state0))
        return controller

class EnergyStabilization(Controller):
    pass
#    def __init__

