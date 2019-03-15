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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('teacher_model_dir', '/docker_data/ModelCompression/model/teacher/', 'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('stu_model_dir', '/docker_data/ModelCompression/model/student_hinton/model.ckpt', 'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_readers', 4, 'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 1, 'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('epoch_num', 1200, 'Epoch number.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'The name of the optimizer, one of "adadelta", "adagrad", "adam","ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float('adadelta_rho', 0.95, 'The decay rate for adadelta.')
tf.app.flags.DEFINE_float('adagrad_initial_accumulator_value', 0.1, 'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5, 'The learning rate power.')
tf.app.flags.DEFINE_float('ftrl_initial_accumulator_value', 0.1, 'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float('ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float('ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string('learning_rate_decay_type', 'fixed', 'Specifies how the learning rate is decayed. One of "fixed", "exponential", or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('end_learning_rate', 0.0001, 'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 1, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which learning rate decays. Note: this flag counts epochs per clone but aggregates per sync replicas. So 1.0 means that each clone will go over full epoch individually, but replicas will go once across all replicas.')
tf.app.flags.DEFINE_bool('sync_replicas', False, 'Whether or not to synchronize the replicas during training.')#是否在培训期间同步副本
tf.app.flags.DEFINE_integer('replicas_to_aggregate', 1, 'The Number of gradients to collect before updating params.')#更新参数之前要收集的梯度数

#######################
# Dataset Flags #
#######################
tf.app.flags.DEFINE_string('dataset_name', 'img_cut', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string('dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string('dataset_split_name2', 'validation', 'The name of the train/test split.')
tf.app.flags.DEFINE_string('dataset_dir', '/docker_data/ModelCompression/data/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('labels_offset', 0, 'An offset for the labels in the dataset. This flag is primarily used to evaluate the VGG and ResNet architectures which do not use a background class for the ImageNet dataset.')
tf.app.flags.DEFINE_string('model_name', 'vgg_16', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string('stu_model_name', 'vgg_16_stu', 'The name of the architecture to train.') # by luckygong
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer('batch_size', 50, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('train_image_size', 224, 'Train image size')

#####################
#   KD Parameters   #
#####################
tf.app.flags.DEFINE_float('lamda', 0.3,'') # by luckygong
tf.app.flags.DEFINE_float('tau', 4.0,'') # by luckygong

FLAGS = tf.app.flags.FLAGS## tf.app.flags.DEFINE_string("param_name", "default_val", "description")

teacher_para_name = ['vgg_16/conv1/conv1_1/weights:0', 
    'vgg_16/conv1/conv1_1/biases:0', 
    'vgg_16/conv1/conv1_2/weights:0', 
    'vgg_16/conv1/conv1_2/biases:0', 
    'vgg_16/conv2/conv2_1/weights:0', 
    'vgg_16/conv2/conv2_1/biases:0', 
    'vgg_16/conv2/conv2_2/weights:0', 
    'vgg_16/conv2/conv2_2/biases:0', 
    'vgg_16/conv3/conv3_1/weights:0', 
    'vgg_16/conv3/conv3_1/biases:0', 
    'vgg_16/conv3/conv3_2/weights:0', 
    'vgg_16/conv3/conv3_2/biases:0', 
    'vgg_16/conv3/conv3_3/weights:0', 
    'vgg_16/conv3/conv3_3/biases:0', 
    'vgg_16/conv4/conv4_1/weights:0', 
    'vgg_16/conv4/conv4_1/biases:0', 
    'vgg_16/conv4/conv4_2/weights:0', 
    'vgg_16/conv4/conv4_2/biases:0', 
    'vgg_16/conv4/conv4_3/weights:0', 
    'vgg_16/conv4/conv4_3/biases:0', 
    'vgg_16/conv5/conv5_1/weights:0', 
    'vgg_16/conv5/conv5_1/biases:0', 
    'vgg_16/conv5/conv5_2/weights:0', 
    'vgg_16/conv5/conv5_2/biases:0', 
    'vgg_16/conv5/conv5_3/weights:0', 
    'vgg_16/conv5/conv5_3/biases:0', 
    'vgg_16/fc6/weights:0', 
    'vgg_16/fc6/biases:0', 
    'vgg_16/fc7/weights:0', 
    'vgg_16/fc7/biases:0', 
    'vgg_16/fc8/weights:0', 
    'vgg_16/fc8/biases:0'
]

student_para_name = ['vgg_16_stu/conv1/conv1_1/weights:0', 
'vgg_16_stu/conv1/conv1_1/biases:0', 
'vgg_16_stu/conv1/conv1_2/weights:0', 
'vgg_16_stu/conv1/conv1_2/biases:0', 
'vgg_16_stu/conv2/conv2_1/weights:0', 
'vgg_16_stu/conv2/conv2_1/biases:0', 
'vgg_16_stu/conv2/conv2_2/weights:0', 
'vgg_16_stu/conv2/conv2_2/biases:0', 
'vgg_16_stu/conv3/conv3_1/weights:0', 
'vgg_16_stu/conv3/conv3_1/biases:0', 
'vgg_16_stu/conv3/conv3_2/weights:0', 
'vgg_16_stu/conv3/conv3_2/biases:0', 
'vgg_16_stu/conv3/conv3_3/weights:0', 
'vgg_16_stu/conv3/conv3_3/biases:0', 
'vgg_16_stu/conv4/conv4_1/weights:0', 
'vgg_16_stu/conv4/conv4_1/biases:0', 
'vgg_16_stu/conv4/conv4_2/weights:0', 
'vgg_16_stu/conv4/conv4_2/biases:0', 
'vgg_16_stu/conv4/conv4_3/weights:0', 
'vgg_16_stu/conv4/conv4_3/biases:0', 
'vgg_16_stu/fc6/weights:0', 
'vgg_16_stu/fc6/biases:0', 
'vgg_16_stu/fc7/weights:0', 
'vgg_16_stu/fc7/biases:0', 
'vgg_16_stu/fc8/weights:0',
'vgg_16_stu/fc8/biases:0']

def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  # Note: when num_clones is > 1, this will actually have each clone to go
  # over each epoch FLAGS.num_epochs_per_decay times. This is different
  # behavior from sync replicas and is expected to produce different results.
  decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                    FLAGS.batch_size)

  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
  return optimizer

def _count_parameter():
    total_parameters = 0
    for variable in tf.all_variables():
        # print(type(variable))
        #print(dir(variable))
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)#设置要记录的消息的阈值

  with tf.Graph().as_default():
    #######################
    #       Config        #
    #######################
    
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    global_step = tf.Variable(0, trainable=False)

    
    print('Config Done.')


    ######################
    # Gen Training Data  #
    ######################

    # select the dataset
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    
    # Select the preprocessing function
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    # Create a dataset provider that loads data from the dataset
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset #offset=0
    image = image_preprocessing_fn(image, FLAGS.train_image_size, FLAGS.train_image_size)  #none
    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=2 * FLAGS.batch_size)
    labels = slim.one_hot_encoding(
        labels, dataset.num_classes - FLAGS.labels_offset)
    print(labels.shape)
    print('Load and Preprocess Training Data Done.')

    #######################
    # Gen Validation Data #
    #######################

    # select the dataset
    dataset2 = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name2, FLAGS.dataset_dir)
    
    # Create a dataset provider that loads data from the dataset
    provider2 = slim.dataset_data_provider.DatasetDataProvider(
        dataset2,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size)

    [image2, label2] = provider.get(['image', 'label'])
    label2 -= FLAGS.labels_offset #offset=0
    image2 = image_preprocessing_fn(image2, FLAGS.train_image_size, FLAGS.train_image_size)  #none
    images2, labels2 = tf.train.batch(
        [image2, label2],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=2 * FLAGS.batch_size)
    labels2 = slim.one_hot_encoding(
        labels2, dataset.num_classes - FLAGS.labels_offset)
    print('Load and Preprocess Validation Data Done.')

 
    #######################
    #   Tea&Stu network   #
    #######################
    # placeholder
    feed_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.train_image_size, FLAGS.train_image_size, 3])
    feed_labels = tf.placeholder(tf.int32, [FLAGS.batch_size, 10])
    loss_labels = tf.argmax(feed_labels, 1)

    # select Teacher network
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),#10-0
        weight_decay=FLAGS.weight_decay,
        is_training=True)
    with slim.arg_scope([slim.conv2d, slim.fully_connected], trainable = False):
      logits, end_points = network_fn(feed_images)
    predictions = tf.argmax(logits, 1)

    # select Student network by luckygong
    stu_network_fn = nets_factory.get_network_fn(
        FLAGS.stu_model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),#10-0
        weight_decay=FLAGS.weight_decay,
        is_training=True)
    stu_logits, stu_end_points = stu_network_fn(feed_images)
    stu_predictions = tf.argmax(stu_logits, 1) # 32
    
    # Define the metrics:#指标
    correct_prediction = tf.equal(loss_labels, stu_predictions)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    correct_prediction = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    ################################
    #            KD Loss           #
    ################################

    # by luckygong

    # Hinton
    teacher_tau = tf.scalar_mul(1.0/FLAGS.tau, logits)
    student_tau = tf.scalar_mul(1.0/FLAGS.tau, stu_logits)
    loss1 = tf.nn.softmax_cross_entropy_with_logits(logits = tf.scalar_mul(FLAGS.tau, student_tau), labels = feed_labels)
    loss2 = tf.scalar_mul(2 * FLAGS.tau * FLAGS.tau, tf.nn.softmax_cross_entropy_with_logits(logits = student_tau, labels = tf.nn.softmax(teacher_tau)))
    tf_loss = (FLAGS.lamda * tf.reduce_sum(loss1) + (1-FLAGS.lamda)*tf.reduce_sum(loss2)) / FLAGS.batch_size

    # NIPS
    # tf_loss = tf.nn.l2_loss(end_points['vgg_16/fc8'] - stu_end_points['vgg_16_stu/fc8']) / FLAGS.batch_size 
    
    print('Set Loss Done.')

    ################################
    #        KD Optimization       #
    ################################

    learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
    optimizer = _configure_optimizer(learning_rate).minimize(tf_loss, global_step = global_step)
    # summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    print('Set Optimization Done.')
    print(_count_parameter())

    ###################
    #  Start Training #
    ###################
    
    with tf.Session(config = config_proto) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess,coord)
      
      teacher_para = []
      for name in teacher_para_name:
        teacher_para.append(tf.get_default_graph().get_tensor_by_name(name))
      tea_saver = tf.train.Saver(teacher_para)
      ckpt = tf.train.get_checkpoint_state(FLAGS.teacher_model_dir)
      if ckpt and ckpt.model_checkpoint_path:
        tea_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.teacher_model_dir))
      print('Definate Network and Load Teacher Model Done')

      student_para = []
      for name in student_para_name:
        student_para.append(tf.get_default_graph().get_tensor_by_name(name))
      stu_saver = tf.train.Saver(student_para)

      for v in tf.all_variables():
          print(v.name)
      
      print('Start Training')
      max_valid_acc = 0
      for i in range(FLAGS.epoch_num):
        # train
        for step in range(int(13000/FLAGS.batch_size)):
          images_this_step, labels_this_step = sess.run([images, labels])
          correct_prediction_this_step, accuracy_this_step, tf_loss_this_step, _, global_step_this_step = sess.run([correct_prediction, accuracy, tf_loss, optimizer, global_step], feed_dict = {feed_images: images_this_step, feed_labels: labels_this_step})
          print('Epoch %d, %d, global step: %d, loss %f, acc %f current count %d' % (i + 1, step + 1, global_step_this_step, tf_loss_this_step, accuracy_this_step, correct_prediction_this_step))
          if(global_step_this_step % 50 == 0):
            # eval
            all_corrent_prediction = 0
            all_prediction = 0
            for eval_step in range(int(500/FLAGS.batch_size)):
              all_prediction += FLAGS.batch_size
              images_this_step, labels_this_step = sess.run([images2, labels2])
              all_corrent_prediction += sess.run([correct_prediction], feed_dict = {feed_images: images_this_step, feed_labels: labels_this_step})[0]
            print('Eval %d, %d, global step:%d, acc %f current count %d, all count %d' % (i + 1, step + 1, global_step_this_step, all_corrent_prediction * 1.0 / all_prediction, all_corrent_prediction, all_prediction))
            if(max_valid_acc < all_corrent_prediction * 1.0 / all_prediction):
              print('Save model, this step acc: %f, last step acc: %f'%(all_corrent_prediction * 1.0 / all_prediction, max_valid_acc))
              max_valid_acc = all_corrent_prediction * 1.0 / all_prediction
              stu_saver.save(sess, FLAGS.stu_model_dir, global_step = global_step_this_step)

if __name__ == '__main__':
    tf.app.run()
