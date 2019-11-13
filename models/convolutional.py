import tensorflow as tf
from models.interfaces import modelABC
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.compat.v1 import variance_scaling_initializer


class LENET_5(modelABC):
    def __init__(self):
        self.input_shape = [None, 32, 32, 1]
        self.output_shape = [None, 84]

    def _cnn(self, features, mode):
        activation = 'relu'
        kernel_initializer = tf.keras.initializers.TruncatedNormal(
            mean=0, stddev=0.1)
        bias_initializer = 'zeros'

        # conv1: output is [None, 28, 28, 6]
        conv1 = Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1),
                       padding='valid', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer)(features)

        # pool1: output is [None, 14, 14, 6]
        pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

        # conv2: output is [None, 10, 10, 16]
        conv2 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),
                       padding='valid', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer)(pool1)

        # pool2: output is [None, 5, 5, 16]
        pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

        return pool2

    def _fcn(self, features, mode):
        activation = 'relu'
        kernel_initializer = tf.keras.initializers.TruncatedNormal(
            mean=0, stddev=0.1)
        bias_initializer = 'zeros'

        # fc3: output is [None, 120]
        fc3 = Dense(units=120, activation=activation, use_bias=True,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)(features)

        # fc4: output is [None, 84]
        fc4 = Dense(units=84, activation=activation, use_bias=True,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)(fc3)

        return fc4

    def model_fn(self, features, mode):
        pool2 = self._cnn(features, mode)
        flatten = Flatten()(pool2)
        fc4 = self._fcn(flatten, mode)

        return fc4
