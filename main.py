import os
import tensorflow as tf
import numpy as np
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('skip-training', False, 'If true, skip training the model.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest check point')

model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

mnist = input_data.read_data_sets('../Data/mnist', one_hot=True)


def weight_bias(W_shape, b_shape, bias_init=0.1):
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b


def dense_layer(x, W_shape, b_shape, activation):
    W, b = weight_bias(W_shape, b_shape)
    return activation(tf.matmul(x, W) + b)


def conv2d_layer(x, W_shape, b_shape, strides, padding):
    W, b = weight_bias(W_shape, b_shape)
    return tf.nn.relu(tf.nn.conv2d(x, W, strides, padding) + b)

def highway_conv2d_layer(x, W_shape, b_shape, strides, padding, carry_bias=-1.0)
    W, b = weight_bias(W_shape, b_shape, carry_bias)
    W_T, b_T = weight_bias(W_shape, b_shape)
    H = tf.nn.relu(tf.nn.conv2d(x, W, strides, padding) + b, name='activation')
    T = tf.sigmoid(tf.nn.conv2d(x, W_T, strides, padding)+b_T, name='transform_gate')
    C = tf.sub(1.0, T, name="carry_gate")
    return tf.add(tf.mul(H,T), tf.mul(x,C), 'y') # y = (H * T) + (x * C)

with tf.Graph().as_default(), tf.Session() as sess:
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])

    carry_bias_init = -1.0

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    keep_prob1 = tf.placeholder("float", name="keep_prob1")
    x_drop = tf.nn.dropout(x_image, keep_prob1)

    prev_y = conv2d_layer(x_drop, [5,5,1,32], [32], [1,1,1,1], 'SAME')
    prev_y = highway_conv2d_layer(prev_y, [3,3,32,32], [32], [1,1,1,1])