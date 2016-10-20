from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class GateLayer(Layer):
    def __init__(self, skip_rate, **kwargs):
        """Return a * squared(x), a is a learnable scalar."""
        self.skip_rate = skip_rate
        super(GateLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        """[:batch] is for conv, [batch:] is for identity."""
        p = np.random.normal(0, 1, (1,))[0]
        zeros = K.zeros_like(x)
        # print x.get_shape().as_list()
        # print x._keras_shape
        # print zeros.get_shape().as_list()

        skip_out = K.concatenate([zeros, x], axis=1)
        no_skip_out = K.concatenate([x, zeros], axis=1)
        if p < self.skip_rate:
            # skip
            return K.in_train_phase(skip_out, skip_out)
        else:
            return K.in_train_phase(no_skip_out, no_skip_out)

    def get_output_shape_for(self, input_shape):
        # input shape: (batch, ...)
        print 'GateLayer: input_shape:', input_shape, \
            'output_shape:', (input_shape[:1]) + (input_shape[1]*2, ) + input_shape[2:]
        return (input_shape[:1]) + (input_shape[1]*2, ) + input_shape[2:]


class SliceLayer(Layer):
    def __init__(self, first_half, **kwargs):
        self.first_half = first_half
        super(SliceLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        num_channels = x.get_shape().as_list()[1]
        assert num_channels % 2 == 0
        middle = num_channels / 2
        return x[:, :middle, :, :] if self.first_half else x[:, middle:, :, :]

    def get_output_shape_for(self, input_shape):
        # print input_shape
        return (input_shape[0], ) + (input_shape[1]/2, ) + input_shape[2:]
