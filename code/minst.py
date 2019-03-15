import tensorflow as tf

import os

output = 'output'

output_dir = os.path.join(os.path.abspath(output), 'weights')

ckpt_file = os.path.join(output_dir, 'save.ckpt')

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])

W = tf.get_variable("W", initializer=tf.zeros([784, 10]), collections=[tf.GraphKeys.GLOBAL_VARIABLES])

b = tf.get_variable("b", initializer=tf.zeros([10]), collections=[tf.GraphKeys.GLOBAL_VARIABLES])

# b1=tf.get_variable("b1",initializer=tf.zeros([10]),collections=[tf.GraphKeys.LOCAL_VARIABLES])


y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

variable_to_restore = tf.global_variables()  # +tf.local_variables()

saver = tf.train.Saver(variable_to_restore, max_to_keep=None)

is_train = False  # True就训练，False为检测
with tf.variable_scope("weights", reuse=True):
    if is_train:

        for i in range(1000):

            batch_xs, batch_ys = mnist.train.next_batch(100)

            train_step.run({x: batch_xs, y_: batch_ys})

            if i % 100 == 0:
                print('Saving checkpoint file to: {}'.format(output_dir))

                saver.save(sess, ckpt_file)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
    else:
        model_file = tf.train.latest_checkpoint(output_dir)
        saver.restore(sess, model_file)
        print(sess.run(W))  # 这里是把之前保存的变量取出来观察一下
        print(sess.run(b))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
