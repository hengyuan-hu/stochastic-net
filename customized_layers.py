from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class GateLayer(Layer):
    def __init__(self, skip_rate, **kwargs):
        """Return a * squared(x), a is a learnable scalar."""
        self.skip_rate = K.variable(np.array([skip_rate]))
        super(GateLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        """[:batch] is for conv, [batch:] is for identity."""
        p = K.random_uniform((1,))
        zeros = K.zeros_like(x)

        skip_out = K.concatenate([zeros, x], axis=1)
        no_skip_out = K.concatenate([x, zeros], axis=1)
        return K.switch(K.lesser(p[0], self.skip_rate[0]),
                        K.in_train_phase(skip_out, no_skip_out),
                        K.in_train_phase(no_skip_out, no_skip_out))

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
