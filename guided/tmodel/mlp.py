import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

class MLP(object):
    def relu(self, x):
        return T.maximum(0.0, x)

    def dropout(self, x, dropout):
        """p is the probablity of dropping a unit
        """
        
        mask = self.srng.binomial(n=1, p=1-dropout, size=x.shape)
        y = x * T.cast(mask, floatX)
        return y

    def make_layer(self, layer_no, in_size, out_size):
            initial_data = (np.random.standard_normal([out_size, in_size])*1.0/out_size).astype(floatX)
            linear_net = theano.shared(initial_data, borrow=True, name='linear_%d' % (layer_no,))
            initial_data = (np.random.standard_normal([out_size])*1.0/out_size).astype(floatX)
            linear_bias = theano.shared(initial_data, borrow=True, name='linear_bias_%d' % (layer_no,))
            return (linear_net, linear_bias)

    def __init__(self, layers, dropout=0.0):
        self.params = []

        self.srng = theano.tensor.shared_randomstreams.RandomStreams(1234)

        self.activations  = []
        self.num_layers   = len(layers) - 1
        self.layers       = layers
        self.dropout_prob = dropout

        for index, (in_layer, out_layer) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            linear, bias = self.make_layer(index, in_layer, out_layer)

            # store the parameters for this layer:
            self.params.append((linear, bias))

            # store what activation is needed:
            if index == 0:
                self.activations.append('sigmoid')
            elif index == len(layers)-2:
                self.activations.append('tanh')
            else:
                self.activations.append('relu')

        self.x                  = T.vector()
        self.prediction         = self.predict(self.x)
        self.prediction_dropout = self.predict_dropout(self.x)

    def activation_function_for_name(self, name):
        if name == 'sigmoid':
            return T.nnet.sigmoid
        elif name == 'tanh':
            return T.tanh
        elif name == 'relu':
            return self.relu
        else:
            raise ValueError("No activation function for %s" % name)

    def predict(self, x):
        y = x
        activation_function = lambda x: x
        for index in range(self.num_layers):
            linear, bias = self.params[index]
            activation_function = self.activation_function_for_name(self.activations[index])
            if self.dropout_prob > 0 and index>0:
                y = y * (1. - self.dropout_prob)
            y = activation_function(T.dot(linear, y) + bias)

        return y

    def predict_dropout(self, x):
        y = x
        activation_function = lambda x: x
        for index in range(self.num_layers):
            linear, bias = self.params[index]
            activation_function = self.activation_function_for_name(self.activations[index])
            if self.dropout_prob > 0 and index>0:
                y = self.dropout(y, self.dropout_prob)
            y = activation_function(T.dot(linear, y) + bias)

        return y

    def predict_dropout_with_mask(self, x, masks):
        y = x
        activation_function = lambda x: x
        for index in range(self.num_layers):
            linear, bias = self.params[index]
            activation_function = self.activation_function_for_name(self.activations[index])
            if self.dropout_prob > 0 and index>0:
                y = y * masks[index-1]
            y = activation_function(T.dot(linear, y) + bias)

        return y

    def get(self):
        return (self.params, self.x, self.prediction, self.prediction_dropout)
