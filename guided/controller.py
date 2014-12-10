import control
import numpy as np

class Controller(object):
    def __init__(self, controller):
        self.controller = controller

    def get(self):
        return self.controller

class LQR(Controller):
    def __init__(self, plant, state0, t0, Q=None, R=None):
        A = plant.df(state0, t0)
        B = plant.control_matrix(state0, t0)
        Q = Q or np.identity(A.shape[1]) * 1.0
        R = R or np.identity(B.shape[1]) * 0.00000001

        K, S, E = control.lqr(A, B, Q, R)
        self.state0 = state0.copy()
        self.K = K
        self.S = S
        self.E = E

    def score(self, state, t):
        xbar = (state - self.state0)[:, np.newaxis]
        score =  np.dot(np.dot(xbar.T, self.S), xbar)
        return score

    def get(self):
        controller =  lambda state, t: np.dot(- self.K, (state - self.state0))
        return controller

# WARNING: this controller works only for double pendulum.
# not sure how to generalize
class EnergyStabilization(Controller):
    def __init__(self, plant, state0, t0, coefficient=1.0):
        self.e_desired = plant.energy(state0, t0)
        self.e_desired = self.e_desired
        self.plant = plant
        self.coefficient = coefficient

    def get(self):
        controller = lambda state, t: self.coefficient * (self.e_desired - self.plant.energy(state, t)) * state[3]
        return controller


class MixedController(Controller):
    def __init__(self):
        self.c = []
        self.regional = []

    def add(self, weight, controller):
        self.c.append((weight, controller.get()))

    def get(self):
        def controller(state, t):
            res = np.array([0.0])

            for in_region, controller in self.regional:
                if in_region(state, t):
                    return controller(state, t)

            for w, controller in self.c:
                output = controller(state, t)
                if res is None:
                    res = w*output
                else:
                    res += w*output
            return res

        return controller

    def add_regional(self, in_region, controller):
        """WARNING: controllers are tested in order they are added.
                    are your regions disjoint?
        """
        self.regional.append((in_region, controller.get()))

    def add_lqr(self, lqr, maximum_score):
        """Activates lqr when it's score is less than maximum_score"""
        self.add_regional(lambda state, t: lqr.score(state, t) < maximum_score,
                          lqr)
