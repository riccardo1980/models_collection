import tensorflow as tf
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.compat.v1.keras import initializers
from typing import List


class alexnet():
    def __init__(self):
        self.INPUT_SHAPE = (None, 227, 227, 3)
        self.OUTPUT_SHAPE = (None, 4096)

    def model_fn(self,
                 features: tf.Tensor,
                 mode: tf.estimator.ModeKeys) -> tf.Tensor:
        """
            alexnet image classification convolutional network

            :param features: input of the network
            :param mode: standard names for Estimator model modes
            :return: output of the network (except last FC that evaluates the logits)

        """

        assert_input_op = tf.debugging.assert_equal(features.get_shape().as_list()[1:],
                                                    self.INPUT_SHAPE[1:])
        with tf.control_dependencies([assert_input_op]):
            pool5 = self._cnn(features, mode)

        flatten = Flatten()(pool5)
        drop8 = self._fcn(flatten, mode)

        assert_output_op = tf.debugging.assert_equal(drop8.get_shape().as_list()[1:],
                                                     self.OUTPUT_SHAPE[1:])
        with tf.control_dependencies([assert_output_op]):
            drop8 = tf.identity(drop8)

        return drop8

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

        conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                       padding='valid', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(features)

        pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                          padding='valid')(conv1)

        conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
                       padding='same', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(pool1)

        pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                          padding='valid')(conv2)

        conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                       padding='same', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(pool2)

        conv4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                       padding='same', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(conv3)

        conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                       padding='same', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(conv4)

        pool5 = MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                          padding='valid')(conv5)

        return pool5

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
        kernel_initializer = initializers.TruncatedNormal(mean=0, stddev=0.1)
        bias_initializer = 'zeros'

        fc6 = Dense(units=496, activation=activation, use_bias=True,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)(features)

        drop7 = Dropout(rate=0.5)(fc6)

        fc7 = Dense(units=496, activation=activation, use_bias=True,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)(drop7)

        drop8 = Dropout(rate=0.5)(fc7)

        return drop8

    def get_feature_columns(self) -> List[object]:
        feature_columns = [
            tf.feature_column.numeric_column('images', self.INPUT_SHAPE[1:]),
        ]
        return feature_columns
