from .multi_step_policy_model import MultiStepPolicyModel
import theano, theano.tensor as T
import numpy as np
from numpy import pi
from .rnn_mlp import RNNMLP


class TemporalMultiStepPolicyModel(MultiStepPolicyModel):
    """
    Main changes in this policy model are the addition of:

    1) a recurrent neural network instead of its non-recurrent MLP version
    
    2) a step forward function with hidden states

    3) new update and prediction functions that use the hidden states and
       propagate those forward.
    
    """

    def __init__(self, hidden_size = 20, **kwargs):
        self.hidden_size = hidden_size
        self.predict_with_hidden = {}
        super().__init__(**kwargs)

    def create_variables(self):
        self.params = []
        self.acquire_RNN_MLP()

    def acquire_RNN_MLP(self):
        self._mlp = RNNMLP(layers= [self.state_size] + self.internal_layers + [self.policy_size * 2],
                    hidden_size = self.hidden_size,
                    dropout=self.dropout, softmax = True)
        mlp_params, mlp_x, mlp_prediction, mlp_prediction_dropout = \
                self._mlp.get()
        for param_set in mlp_params:
            self.params.extend(param_set)
        
        self._prediction = self.boundsify(mlp_prediction[0])
        self._prediction_dropout = self.boundsify(mlp_prediction_dropout[0])
        self._x = mlp_x

    def step_forward(self, position, hiddens):
        """
        Step forward in time using the policy
        and the known system dynamics.

        """
        policy, new_hiddens = self._mlp.predict_dropout(position, hiddens)

        u = self.boundsify(policy)
        new_position = self.dynamics( position, u )

        return (new_position, new_hiddens, policy)

    def controller(self):
        def c(x,t):
            if any(np.isnan(x)) or not all(np.abs(x) < 1e100):
                return 0.0

            if not hasattr(self, '_current_hiddens'):
                # start off with biases:
                self._current_hiddens = [params[3].get_value() for params in self._mlp.params]

            prediction, *hiddens = self.predict_with_hidden['test'](np.array(x), *self._current_hiddens)
            self._current_hiddens = hiddens
            return prediction
        return c

    def create_misc_functions(self, name, prediction):
        self.predict[name] = theano.function([self._x],
                                             prediction,
                                             allow_input_downcast=self.allow_input_downcast)

        hiddens = [T.vector() for i in range(self._mlp.params)]
        self.predict_with_hidden[name] = theano.function([self._x, hiddens],
                                             self._mlp.predict(self._x, hiddens),
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
        new_hiddens  = self._mlp.initial_hiddens()

        for i in range(self.num_steps):
            (new_position, new_hiddens, policy) = self.step_forward(new_position, new_hiddens)

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