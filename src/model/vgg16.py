# coding=utf-8
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.

      Args:
        weight_decay: The l2 regularization coefficient.

      Returns:
        An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

def vgg_16(num_classes=10, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=False, scope='vgg_16', fc_conv_padding='VALID', global_pool=False):
    """Oxford Net VGG 16-Layers version D Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

      Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes. If 0 or None, the logits layer is
          omitted and the input features to the logits layer are returned instead.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
          layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
          outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.
        fc_conv_padding: the type of padding to use for the fully connected layer
          that is implemented as a convolutional layer. Use 'SAME' padding if you
          are applying the network in a fully convolutional manner and want to
          get a prediction map downsampled by a factor of 32 as an output.
          Otherwise, the output prediction map will be (input / 32) - 6 in case of
          'VALID' padding.
        global_pool: Optional boolean flag. If True, the input to the classification
          layer is avgpooled to size 1x1, for any input size. (This is not part
          of the original VGG architecture.)

      Returns:
        net: the output of the logits layer (if num_classes is a non-zero integer),
          or the input to the logits layer (if num_classes is 0 or None).
        end_points: a dict of tensors with intermediate activations.
    """
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x_placeholder')
    y = tf.placeholder(tf.int64, shape=[None, num_classes], name='y_placeholder')
    with tf.variable_scope(scope, 'vgg_16', [x]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d], outputs_collections=end_points_collection):
            net = slim.repeat(x, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            shape = int(np.prod(net.get_shape()[1:]))
            net = slim.fully_connected(tf.reshape(net, [-1, shape]), 4096, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

            # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            
            # if global_pool:
            #   net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            #   end_points['global_pool'] = net

            # net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
            net = slim.fully_connected(net, num_classes, scope='fc8')

            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            
            finaloutput = tf.nn.softmax(net, name="softmax")
            print('finaloutput.shape ' + str(finaloutput.shape))
            prediction_labels = tf.argmax(finaloutput, axis=1, name="output")
            print('prediction_labels.shape ' + str(prediction_labels.shape))
            print('y_labels.shape ' + str(tf.argmax(y, axis = 1).shape))

            end_points['loss'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
            end_points['optimizer'] = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(end_points['loss'])
            end_points['accuracy'] = tf.reduce_mean(tf.cast(tf.equal(prediction_labels, tf.argmax(y, axis = 1)), tf.float32))
            end_points['correct_times'] = tf.reduce_sum(tf.cast(tf.equal(prediction_labels, tf.argmax(y, axis = 1)), tf.int32))

            end_points['x'] = x
            end_points['y'] = y

            return net, end_points

vgg_16.default_image_size = 224
vgg_d = vgg_16