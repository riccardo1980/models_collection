import tensorflow as tf
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.compat.v1.keras import initializers
from typing import List

INPUT_SHAPE = (None, 227, 227, 3)
OUTPUT_SHAPE = (None, 4096)

# [TODO] design an uniform API:
# something like: def model_fn(features, labels, mode, params)

def model_fn(features, 
             activation = 'relu',
             kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.1),
             bias_initializer = 'zeros'):

    assert_input_op = tf.debugging.assert_equal(features.get_shape().as_list()[1:],
                                                INPUT_SHAPE[1:])

    # input: [None, 227, 227, 3]
    # conv1: f 96, k (11,11), s (4,4), VALID, relu --> [None, 54, 54, 96]
    with tf.control_dependencies([assert_input_op]):
        conv1 = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), 
                       padding='valid', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(features)

    # pool1: k (3,3), s (2,2), VALID               --> [None, 26, 26, 96]
    with tf.control_dependencies(tf.debugging.assert_equal(conv1.get_shape()[1:], [54,54,96])):
        pool1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(conv1)

    # conv2: f 256, k (5,5), s (1,1), SAME, relu   --> [None, 26, 26, 256]
    with tf.control_dependencies(tf.debugging.assert_equal(features.get_shape()[1:], [26,26,96])):
        conv2 = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), 
                       padding='same', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(pool1)

    # pool2: k (3,3), s (2,2), VALID               --> [None, 12, 12, 256]
    with tf.control_dependencies(tf.debugging.assert_equal(conv1.get_shape()[1:], [26,26,256])):
        pool2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(conv2)

    # conv3: f 384, k (3,3), s(1,1), SAME, relu    --> [None, 12, 12, 384]
    with tf.control_dependencies(tf.debugging.assert_equal(features.get_shape()[1:], [12,12,256])):
        conv3 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), 
                       padding='same', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(pool2)

    # conv4: f 384, k (3,3), s(1,1), SAME, relu    --> [None, 12, 12, 384]
    with tf.control_dependencies(tf.debugging.assert_equal(features.get_shape()[1:], [12,12,384])):
        conv4 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), 
                       padding='same', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(conv3)

    # conv5: f 256, k (3,3), s(1,1), SAME, relu    --> [None, 12, 12, 256]
    with tf.control_dependencies(tf.debugging.assert_equal(features.get_shape()[1:], [12,12,384])):
        conv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), 
                       padding='same', activation=activation, use_bias=True,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(conv4)

    # pool5: k (3,3), s (2,2)                      --> [None,  5,  5, 256]
    with tf.control_dependencies(tf.debugging.assert_equal(conv1.get_shape()[1:], [12,12,256])):
        pool5 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(conv5)

    # flatten --> [None, 6400]
    flatten = Flatten()(pool5)

    # fc6: f 4096, relu --> [None, 4096]
    with tf.control_dependencies(tf.debugging.assert_equal(flatten.get_shape()[1:], [6400])):
        fc6 = Dense(units=496, activation=activation, use_bias=True,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)(flatten)

    # drop7: p 0.5      --> [None, 4096]
    drop7 = Dropout(rate=0.5)(fc6)

    # fc7: f 4096, relu --> [None, 4096]
    with tf.control_dependencies(tf.debugging.assert_equal(fc6.get_shape()[1:], [6400])):
        fc7 = Dense(units=496, activation=activation, use_bias=True,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)(drop7)
    
    # drop8: p 0.5      --> [None, 4096]
    drop8 = Dropout(rate=0.5)(fc7)

    return drop8