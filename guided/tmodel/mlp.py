import theano.tensor as T
from .utils import make_layer, activation_function_for_name, dropout as dropout_fun


class MLP(object):

    def __init__(self, layers = [], dropout=0.0, softmax = False):

        self.params = []
        self.activations  = []
        self.softmax = softmax
        self.num_layers   = len(layers) - 1
        self.layers       = layers
        self.dropout_prob = dropout

        self.create_variables()
        self.create_predictions()

    def create_predictions(self):
        self.x                  = T.vector()
        self.prediction         = self.predict(self.x)
        self.prediction_dropout = self.predict_dropout(self.x)

    def create_variables(self):
        for index, (in_layer, out_layer) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            linear, bias = make_layer(index, in_layer, out_layer)

            # store the parameters for this layer:
            self.params.append((linear, bias))

            # store what activation is needed:
            if index == 0:
                self.activations.append('linear')
            elif index == len(self.layers)-2:
                self.activations.append('softmax' if self.softmax else 'tanh')
            else:
                self.activations.append('relu')

    def predict(self, x, dropout = False):
        """
        Predict from the inputs the state of the system.

        """
        y = x
        activation_function = lambda x: x
        for index in range(self.num_layers):
            linear, bias = self.params[index]
            activation_function = activation_function_for_name(self.activations[index])
            if self.dropout_prob > 0 and index>0:
                if dropout:
                    y = dropout_fun(y, self.dropout_prob)
                else:
                    y = y * (1. - self.dropout_prob)
            y = activation_function(T.dot(linear, y) + bias)

        return y

    def predict_dropout(self, x):
        """
        Predict from the inputs the state of the system.

        """
        return self.predict(x, dropout = True)

    def get(self):
        """
        Retrieve the key parameters from the model.

        """
        return (self.params, self.x, self.prediction, self.prediction_dropout)
