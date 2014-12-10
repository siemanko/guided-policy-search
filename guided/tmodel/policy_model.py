import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX
from collections import OrderedDict

from guided.tmodel import MLP

class PolicyModel(object):
    """
    Use a Multi Layer Perceptron to discover a
    policy for a dynamics by gradient descending
    over a single timestep.
    
    """
    @classmethod
    def from_plant(cls, plant, **kwargs):
        return cls(state_size=plant.num_states,
                           policy_size=plant.num_controls,
                           policy_bounds=plant.control_bounds,
                           dynamics=plant.theano_dynamics,
                           **kwargs)

    def __init__(self, state_size = 2,
                       policy_size = 1,
                       policy_bounds = (-40,40),
                       dynamics = None,
                       policy_laziness = 0.0000001,
                       internal_layers = [],
                       learning_rate = 0.01,
                       dropout = 0.0,
                       allow_input_downcast=True):
        # constatants
        if dynamics is None:
          raise ValueError("Dynamics must be specified")
        self.dynamics = dynamics
        self.state_size = state_size
        self.policy_size = policy_size
        self.policy_bounds = policy_bounds
        self.internal_layers = internal_layers
        self.allow_input_downcast = allow_input_downcast
        self.learning_rate = theano.shared(np.float64(learning_rate).astype(floatX), name='learning_rate')
        self.dropout = dropout

        self.policy_laziness = theano.shared(np.float64(policy_laziness).astype(floatX), name='policy_laziness')

        self.predict = {}
        self.teleportation = {}
        self.cost = {}
        self.target_value = {}
        self.cost_fun = {}
        # intialization functions
        self.create_variables()
        self.create_misc_functions('test', self._prediction)
        self.create_misc_functions('train', self._prediction_dropout)
        self.create_update_fun()

    def boundsify(self, net_output):
        zero_one = (net_output[0] + 1.0) / 2.0
        u_min, u_max = self.policy_bounds
        return u_min + zero_one * (u_max-u_min)

    def acquire_MLP(self):
        """
        Our MLP uses a tanh to decide on the control input of the system.

        """
        self._mlp = MLP(layers = [self.state_size] + self.internal_layers + [self.policy_size],
                    dropout=self.dropout)
        mlp_params, mlp_x, mlp_prediction, mlp_prediction_dropout = \
                self._mlp.get()
        for param_set in mlp_params:
            self.params.extend(param_set)
        
        self._prediction = self.boundsify(mlp_prediction)
        self._prediction_dropout = self.boundsify(mlp_prediction_dropout)
        self._x = mlp_x

    def create_variables(self):
        self.params = []
        self.acquire_MLP()

    def create_misc_functions(self, name, prediction):
        self.predict[name] = theano.function([self._x],
                                             prediction,
                                             allow_input_downcast=self.allow_input_downcast)

        self.target_value[name] = T.vector()
        new_x = self.dynamics( self._x, prediction )
        self.teleportation[name] = theano.function([self._x],
                                                   new_x,
                                                   allow_input_downcast=self.allow_input_downcast)

        self.cost[name] = T.sum((new_x - self.target_value[name])**2)
        self.cost_fun[name] = theano.function([self._x, self.target_value[name]],
                                              self.cost[name],
                                              allow_input_downcast=self.allow_input_downcast)

    def create_update_fun(self):
        gparams = T.grad(self.cost['train'], self.params)

        self.grad_fun = theano.function([self._x, self.target_value['train']],
                                        gparams,
                                        allow_input_downcast=self.allow_input_downcast)

        updates=OrderedDict()
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * self.learning_rate

        self.update_fun = theano.function([self._x, self.target_value['train']],
                                          self.cost['train'],
                                          allow_input_downcast=self.allow_input_downcast,
                                          updates = updates)
    def set_learning_rate(self, rate):
        self.learning_rate.set_value(np.float32(rate))

    def reset_weights(self):
        for param in self.params:
            param.set_value(
                (np.random.standard_normal(param.get_value(borrow=True).shape) *
                (1./param.get_value(borrow=True).shape[0])).astype(floatX))

    def controller(self):
        def c(x,t):
            if any(np.isnan(x)) or not all(np.abs(x) < 1e100):
                return 0.0
            prediction = self.predict['test'](np.array(x))
            return prediction
        return c
