from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class EntryGateLayer(Layer):
    def __init__(self, switch_rate, **kwargs):
        self.switch_rate = K.variable(np.array([switch_rate]))
        super(EntryGateLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        p = K.random_uniform((1,))
        zeros = K.zeros_like(x)

        left = K.concatenate([zeros, x], axis=1)
        right = K.concatenate([x, zeros], axis=1)
        both = K.concatenate([x,x],axis=1)
        return K.switch(K.lesser(p[0], self.switch_rate[0]),
                        K.in_train_phase(left, both),
                        K.in_train_phase(right, both))

    def get_output_shape_for(self, input_shape):
        # input shape: (batch, ...)
        print 'GateLayer: input_shape:', input_shape, \
            'output_shape:', (input_shape[:1]) + (input_shape[1]*2, ) + input_shape[2:]
        return (input_shape[:1]) + (input_shape[1]*2, ) + input_shape[2:]

class ExitGateLayer(Layer):
    def __init__(self, **kwargs):
        super(ExitGateLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # return K.in_train_phase(x, x/2)
        return K.in_train_phase(x, x)

    def get_output_shape_for(self, input_shape):
        # identity mapping
        return input_shape


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
