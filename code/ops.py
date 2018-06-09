import tensorflow as tf
import numpy as np

def pooling(input, name, k_size, stride, mode):
    with tf.variable_scope(name):
        return tf.nn.pool(input=input,
                          window_shape=[1, k_size[0], k_size[1], 1],
                          pooling_type=mode,
                          padding='SAME',
                          strides=[1, stride[0], stride[1], 1])

def conv2d(input, name, num_filters, filter_size, stride, reuse, pad='SAME'):
    with tf.variable_scope(name, reuse=reuse):
        stride_shape = [1, stride, stride, 1]
        filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]

        w = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
        conv = tf.nn.conv2d(input, w, stride_shape, padding=pad)
        b = tf.get_variable('b', [1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
        conv = conv + b
        return conv

def conv2d_transpose(input, name, num_filters, filter_size, stride, reuse, pad='SAME'):
    with tf.variable_scope(name, reuse=reuse):
#        n, h, w, c = input.get_shape().as_list()
#        stride_shape = [1, stride, stride, 1]
#        filter_shape = [filter_size, filter_size, num_filters, c]
#        output_shape = [n, int(h*stride), int(w*stride), int(num_filters)]
#
#        w = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
#        deconv = tf.nn.conv2d_transpose(input, w, output_shape, stride_shape, pad)
#        b = tf.get_variable('b', [1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
#        deconv = deconv + b
        deconv = tf.layers.conv2d_transpose(inputs=input, filters=num_filters, kernel_size=(filter_size, filter_size), strides=(stride, stride), padding='same')
        return deconv

def residual(input, name, num_filters, reuse, pad='SAME'):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d(input, 'res1', num_filters, 3, 1, reuse, pad)
        out = tf.nn.relu(out)
        out = conv2d(out, 'res2', num_filters, 3, 1, reuse, pad)
        return out + input

def prelu(input, name):
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', [1], tf.float32, tf.constant_initializer(0.0))
        return 0.5 * (input + tf.abs(input)) + 0.5 * alpha * (input - tf.abs(input))
