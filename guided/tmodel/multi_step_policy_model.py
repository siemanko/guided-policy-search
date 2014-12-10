from .policy_model import PolicyModel
import theano, theano.tensor as T
import numpy as np
from numpy import pi
from .mlp import MLP

class MultiStepPolicyModel(PolicyModel):
    """
    Use a Multi Layer Perceptron to discover a
    policy for a dynamics by gradient descending
    over multiple timesteps.

    """
    
    def __init__(self, num_steps = 10, penalize_over_trajectory = True, mod_by_pi=False, **kwargs):
        self.num_steps = num_steps
        self.penalize_over_trajectory = penalize_over_trajectory
        self.mod_by_pi = mod_by_pi
        super().__init__(**kwargs)

    def step_forward(self, position):
        """
        Step forward in time using the policy
        and the known system dynamics.

        """
        policy = self._mlp.predict_dropout(position)
        u = self.boundsify(policy)
        new_position = self.dynamics( position, u )

        return (new_position, policy)

    def acquire_MLP(self):
        """
        Our MLP has a softmax instead of a tanh, we then convert the output of the policy
        into a bounded scalar.

        """
        self._mlp = MLP(layers = [self.state_size] + self.internal_layers + [self.policy_size * 2],
                    dropout=self.dropout, softmax = True)
        mlp_params, mlp_x, mlp_prediction, mlp_prediction_dropout = \
                self._mlp.get()
        for param_set in mlp_params:
            self.params.extend(param_set)
        
        self._prediction = self.boundsify(mlp_prediction)
        self._prediction_dropout = self.boundsify(mlp_prediction_dropout)
        self._x = mlp_x

    def boundsify(self, net_output):
        u_min, u_max = self.policy_bounds
        return (u_min * net_output[0]) + (u_max * net_output[1])

    def create_misc_functions(self, name, prediction):
        self.predict[name] = theano.function([self._x],
                                             prediction,
                                             allow_input_downcast=self.allow_input_downcast)

        
        new_x = self.dynamics( self._x, prediction )
        self.teleportation[name] = theano.function([self._x],
                                                   new_x,
                                                   allow_input_downcast=self.allow_input_downcast)

        # now do multi step cost:
        self.target_value[name] = T.vector()

        # this sequence controls how many time steps are modeled by the system
        # and get backproped to.
        cost = np.float32(0.0).astype(theano.config.floatX)

        # loop through and update cost
        new_position = self._x
        for i in range(self.num_steps):
            (new_position, policy) = self.step_forward(new_position)
            if self.mod_by_pi:
              new_position = T.set_subtensor( new_position[0], new_position[0] % (2 * pi))
              new_position = T.set_subtensor( new_position[2], new_position[2] % (2 * pi))

            if self.penalize_over_trajectory:
                cost += ((new_position - self.target_value[name])**2).sum()
            cost += (abs(policy)).sum() * self.policy_laziness

        if not self.penalize_over_trajectory:
            cost += ((new_position - self.target_value[name])**2).sum()

        self.cost[name] = cost
        self.cost_fun[name] = theano.function([self._x, self.target_value[name]],
                                             self.cost[name],
                                             allow_input_downcast=self.allow_input_downcast)