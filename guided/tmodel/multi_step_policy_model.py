from .policy_model import PolicyModel
import theano, theano.tensor as T
from collections import OrderedDict
from .mlp import MLP

class MultiStepPolicyModel(PolicyModel):
    """
    Use a Multi Layer Perceptron to discover a
    policy for a dynamics by gradient descending
    over multiple timesteps.
    """

    def step_forward(self, i, position, *masks):
        """
        Step forward in time using the policy
        and the known system dynamics.

        """
        policy = self._mlp.predict_dropout_with_mask(position, [mask[i] for mask in masks])
        u = self.boundsify(policy)
        new_position = self.dynamics(position, u )[0]

        return [new_position, policy]

    def create_misc_functions(self, name, prediction):
        self.predict[name] = theano.function([self._x],
                                             prediction,
                                             allow_input_downcast=self.allow_input_downcast)

        
        new_x = self.dynamics(   self._x, prediction )
        self.teleportation[name] = theano.function([self._x],
                                                   new_x,
                                                   allow_input_downcast=self.allow_input_downcast)

        # now do multi step cost:
        self.target_value[name] = T.vector()

        outputs_info = [
            dict(initial = self._x, name='position', taps=[-1]),
            dict(name="policy", taps=[])
            ]
        # this sequence controls how many time steps are modeled by the system
        # and get backproped to.


        self.num_steps = T.iscalar()
        indices = T.arange(0, self.num_steps)

        masks = [T.cast(self._mlp.srng.binomial(n=1, p=1-self.dropout, size=(self.num_steps, in_layer)), theano.config.floatX) for index, (in_layer, out_layer) in enumerate(zip(self._mlp.layers[:-1], self._mlp.layers[1:])) if index > 0]

        result, updates = theano.scan(self.step_forward,
            sequences         = [ indices ],
            non_sequences     = [ ] + masks,
            outputs_info      = outputs_info)

        self.teleportation_steps = theano.function([self._x, self.num_steps],
                                                   result[-1],
                                                   allow_input_downcast=self.allow_input_downcast)

        # optimize for all timesteps:
        self.cost[name] = T.sum((result[0] - self.target_value[name])**2) + T.sum(result[1] ** 2) * self.policy_laziness
        self.cost_fun[name] = theano.function([self._x, self.target_value[name], self.num_steps],
                                              self.cost[name],
                                              allow_input_downcast=self.allow_input_downcast)

    def create_update_fun(self):
        gparams = T.grad(self.cost['train'], self.params)

        self.grad_fun = theano.function([self._x, self.target_value['train'], self.num_steps],
                                        gparams,
                                        allow_input_downcast=self.allow_input_downcast)

        updates=OrderedDict()
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * self.learning_rate



        self.update_fun = theano.function([self._x, self.target_value['train'], self.num_steps],
                                          self.cost['train'],
                                          allow_input_downcast=self.allow_input_downcast,
                                          updates = updates)

