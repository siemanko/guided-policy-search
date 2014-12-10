import theano, theano.tensor as T
import numpy as np
srng = theano.tensor.shared_randomstreams.RandomStreams(1234)

def relu(x):
    return T.maximum(0.0, x)

def softrelu(x):
    return T.log1p(T.exp(x))

def dropout(x, dropout):
    """p is the probablity of dropping a unit
    """
    
    mask = srng.binomial(n=1, p=1-dropout, size=x.shape)
    y = x * T.cast(mask, theano.config.floatX)
    return y

def activation_function_for_name(name):
    if name == 'sigmoid':
        return T.nnet.sigmoid
    elif name == 'tanh':
        return T.tanh
    elif name == 'relu':
        return relu
    elif name == 'softrelu':
        return softrelu
    elif name == "softmax":
        return lambda x: T.nnet.softmax(x)[0]
    elif name == 'linear':
        return lambda x: x
    else:
        raise ValueError("No activation function for %s" % name)

# TODO: turn layers into classes for simplicity and readability

def make_layer(layer_no, in_size, out_size):
    initial_data = (np.random.standard_normal([out_size, in_size])*1.0/out_size).astype(theano.config.floatX)
    linear_net = theano.shared(initial_data, borrow=True, name='linear_%d' % (layer_no,))
    initial_data = (np.random.standard_normal([out_size])*1.0/out_size).astype(theano.config.floatX)
    linear_bias = theano.shared(initial_data, borrow=True, name='linear_bias_%d' % (layer_no,))
    return (linear_net, linear_bias)

def make_rnn_layer(layer_no, in_size, out_size, hidden_size):
    # out goes up, so hidden is not affected by it.
    (linear_net, linear_bias) = make_layer(layer_no, in_size + hidden_size, out_size)

    # input + hidden -> hidden
    initial_data = (np.random.standard_normal([hidden_size, in_size + hidden_size])*1.0/hidden_size).astype(theano.config.floatX)
    hidden_forward = theano.shared(initial_data, borrow=True, name='hidden_to_hidden_linear_%d' % (layer_no,))

    initial_data = (np.random.standard_normal([hidden_size])*1.0/hidden_size).astype(theano.config.floatX)
    hidden_bias = theano.shared(initial_data, borrow=True, name='hidden_to_hidden_bias_%d' % (layer_no,))

    return (linear_net, linear_bias, hidden_forward, hidden_bias)