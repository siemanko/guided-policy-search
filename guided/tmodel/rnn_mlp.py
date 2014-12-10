import theano.tensor as T
from .utils import make_rnn_layer, activation_function_for_name, dropout as dropout_fun
from .mlp import MLP

class RNNMLP(MLP):
    """
    Stacked recurrent neural networks, with dropout, for extra modernity and fanciness.
    """
    # for simplicy all layers have the same amount of hidden units.
    def __init__(self, hidden_size = 1, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    def initial_hiddens(self):
        return [hidden_bias for linear, bias, hidden_linear, hidden_bias in self.params]

    def create_predictions(self):
        self.x                  = T.vector()
        self.prediction         = self.predict(self.x, self.initial_hiddens())
        self.prediction_dropout = self.predict_dropout(self.x, self.initial_hiddens())

    def create_variables(self):
        for index, (in_layer, out_layer) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            linear, bias, hidden_linear, hidden_bias = make_rnn_layer(index, in_layer, out_layer, self.hidden_size)

            # store the parameters for this layer:
            self.params.append((linear, bias, hidden_linear, hidden_bias))

            # store what activation is needed:
            if index == 0:
                self.activations.append('linear')
            elif index == len(self.layers)-2:
                self.activations.append('softmax' if self.softmax else 'tanh')
            else:
                self.activations.append('relu')

    def predict(self, x, hiddens, dropout = False):
        """
        Predict from the inputs and hidden units the state of the system.

        Note: Predict now takes the previous hidden states as input

        """
        y = x
        new_hiddens = []
        for index in range(self.num_layers):
            linear, bias, hidden_linear, hidden_bias = self.params[index]
            activation_function = activation_function_for_name(self.activations[index])
            if self.dropout_prob > 0 and index>0:
                if dropout:
                    y = dropout_fun(y, self.dropout_prob)
                else:
                    y = y * (1. - self.dropout_prob)
            # 1) take the unprocessed input and use it to compute a new hidden state:
            obs = T.concatenate([y, hiddens[index]])
            new_hiddens.append( T.tanh( T.dot(hidden_linear, obs ) + hidden_bias ) )
            # 2) take the unprocessed input and use it to compute the input for the next level
            # of the network:
            y = activation_function(T.dot(linear, obs ) + bias)

        return y, new_hiddens

    def predict_dropout(self, x, hiddens):
        """
        Predict from the inputs and hidden units the state of the system.

        Note: Predict now takes the previous hidden states as input

        """
        return self.predict(x, hiddens, dropout = True)
