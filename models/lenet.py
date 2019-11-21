import tensorflow as tf
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.compat.v1.keras import initializers
from typing import List, Dict


class lenet():
    def __init__(self):
        self.INPUT_SHAPE = (None, 32, 32, 1)
        self.OUTPUT_SHAPE = (None, 84)

    def model_fn(self,
                 features: tf.Tensor,
                 mode: tf.estimator.ModeKeys) -> tf.Tensor:
        """
            LENET 5 image classification convolutional network

            :param features: input of the network
            :param mode: standard names for Estimator model modes
            :return: output of the network (except last FC that evaluates the logits)

        """

        assert_input_op = tf.debugging.assert_equal(features.get_shape().as_list()[1:],
                                                    self.INPUT_SHAPE[1:])
        with tf.control_dependencies([assert_input_op]):
            pool2 = self._cnn(features, mode)

        flatten = Flatten()(pool2)
        fc4 = self._fcn(flatten, mode)

        assert_output_op = tf.debugging.assert_equal(fc4.get_shape().as_list()[1:],
                                                     self.OUTPUT_SHAPE[1:])
        with tf.control_dependencies([assert_output_op]):
            fc4 = tf.identity(fc4)

        return fc4

    def _cnn(self,
             features: tf.Tensor,
             mode: tf.estimator.ModeKeys) -> tf.Tensor:
        """
            Feature extractor based on Conv layers

            :param features: input of the sub network
            :param mode: standard names for Estimator model modes
            :return: output of the sub network

        """
        activation = 'relu'
        kernel_initializer = initializers.TruncatedNormal(mean=0, stddev=0.1)
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

    def _fcn(self,
             features: tf.Tensor,
             mode: tf.estimator.ModeKeys) -> tf.Tensor:
        """
            Sequence of FullyConnected Layers

            :param features: input of the sub network
            :param mode: standard names for Estimator model modes
            :return: output of the sub network

        """
        activation = 'relu'
        kernel_initializer = tf.compat.v1.keras.initializers.TruncatedNormal(
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

    def get_feature_columns(self) -> List[object]:
        feature_columns = [
            tf.feature_column.numeric_column('images', self.INPUT_SHAPE[1:]),
        ]
        return feature_columns
