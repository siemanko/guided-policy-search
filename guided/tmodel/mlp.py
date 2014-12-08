import numpy as np
import matplotlib.pyplot as plt
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

        self.x = T.vector()

        self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                1234)

        self.prediction = self.x
        self.prediction_dropout = self.x
        activation_function = lambda x: x
        self.activations = []

        for index, (in_layer, out_layer) in enumerate(zip(layers[:-1], layers[1:])):
            linear, bias = self.make_layer(index, in_layer, out_layer)

            if index == 0:
                activation_function = lambda x : 1.0/(1+T.exp(-x))
                self.activations.append('sigmoid')
            elif index == len(layers)-2:
                activation_function = T.tanh
                self.activations.append('tanh')
            else:
                activation_function = self.relu
                self.activations.append('relu')

            if dropout > 0 and index>0:
                self.prediction_dropout = self.dropout(self.prediction_dropout, dropout)
                self.prediction = self.prediction * (1. - dropout)


            self.prediction = activation_function(T.dot(linear, self.prediction ) + bias)

            self.prediction_dropout = activation_function(T.dot(linear, self.prediction_dropout) + bias)

            self.params.append((linear, bias))

        print self.activations

    def get(self):
        return (self.params, self.x, self.prediction, self.prediction_dropout)
