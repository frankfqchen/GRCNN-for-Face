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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib.slim as slim

def GRCL_one_step(net, f_map, feed_gate_out, scope=None):
 with tf.variable_scope(scope, 'GRCL_rec_', [net], reuse=None):
   with tf.variable_scope(scope, 'GRCL_rg', [net], reuse=None):
    rec_bn = slim.batch_norm(net, scope='GRCL_rg_bn')
    rec_fn = tf.nn.relu(rec_bn)
    rec_s =  slim.conv2d(rec_fn, f_map, 1, normalizer_fn=None, activation_fn=None, biases_initializer= tf.zeros_initializer(), scope='GRCL_rg_conv')
   with tf.variable_scope(scope, 'GRCL_r', [net], reuse=None):
    rec_bn1 = slim.batch_norm(net, scope='GRCL_r_bn1')
    rec_fn1 = tf.nn.relu(rec_bn1)
    rec_s1 =  slim.conv2d(rec_fn1, f_map, 3, scope='GRCL_r_conv1')
    rec_s2 =  slim.conv2d(rec_s1, f_map, 3, normalizer_fn=None, activation_fn=None, scope='GRCL_r_conv2')
   gate_total = rec_s + feed_gate_out
   gates = tf.nn.sigmoid(gate_total)
   net_out = rec_s2 * gates + net
   return net_out

def GRCL(net, T, fmap, scope=None, reuse=None):
 with tf.variable_scope(scope, 'GRCL_', [net], reuse=None):
   with tf.variable_scope(scope, 'GRCL_f', [net], reuse=None):
    feed_bn = slim.batch_norm(net, scope='GRCL_f_bn')
    feed_fn = tf.nn.relu(feed_bn)
    state =  slim.conv2d(feed_fn, fmap, 3, normalizer_fn=None, activation_fn=None, scope='GRCL_f_conv')
   with tf.variable_scope(scope, 'GRCL_fg', [net], reuse=None):
    feedg_bn = slim.batch_norm(net, scope='GRCL_fg_bn')
    feedg_fn = tf.nn.relu(feed_bn)
    feedg_f =  slim.conv2d(feed_fn, fmap, 1, normalizer_fn=None, activation_fn=None, scope='GRCL_fg_conv')
   state = slim.repeat(state, T, GRCL_one_step, f_map = fmap, feed_gate_out =feedg_f)
   return state

def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        activation_fn=tf.nn.relu,
                        biases_initializer=None):
      with slim.arg_scope([slim.batch_norm],
                           decay = 0.995,
                           epsilon = 0.001, 
                           updates_collections=None,
                           variables_collections=[ tf.GraphKeys.TRAINABLE_VARIABLES ],
                         ):
        return GRCNN(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def GRCNN(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None, 
                        scope='GRCNN'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = OrderedDict()
  
    with tf.variable_scope(scope, 'GRCNN', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 64, 3, stride=2, scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2,
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1,
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 128, 3, normalizer_fn=None, activation_fn=None,
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net

                net = GRCL(net, 4, 128)
                end_points['GRCL1'] = net
 
                net = slim.batch_norm(net, scope='GRCL_pool1_bn')
                net = tf.nn.relu(net)
                net = slim.conv2d(net, 256, 3, stride=2, normalizer_fn=None, activation_fn=None,
                                  scope='GRCL_pool1')
                end_points['GRCL_pool1'] = net

                net = GRCL(net, 10, 256)
                end_points['GRCL2'] = net

                net = slim.batch_norm(net, scope='GRCL_pool2_bn')
                net = tf.nn.relu(net)
                net = slim.conv2d(net, 512, 3, stride=2, normalizer_fn=None, activation_fn=None,
                                  scope='GRCL_pool2')
                end_points['GRCL_pool2'] = net

                net = GRCL(net, 5, 512)
                net = slim.batch_norm(net, scope='GRCL_final_bn')
                net = tf.nn.relu(net)
                end_points['GRCL3'] = net
                
                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    net = slim.flatten(net)
                    end_points['PreLogitsFlatten'] = net

                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False)
                
                for k, v in end_points.items():
                    print(k,v)
    return net, end_points
