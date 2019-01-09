# coding=utf-8

import sys
sys.path.append('../model')
sys.path.append('../loaddata')
import vgg16
import tensorflow as tf

from data_loader import Loader

slim = tf.contrib.slim

if __name__ == '__main__':

	# load data
    input_file = '/luckygong/data/train-dic3.txt'
    delimiter = '\t'
    raw_size = [224, 224, 3]
    processed_size = [224, 224, 3]
    num_prefetch = 32
    is_training = True
    batch_size = 32
    num_threads = 1
    num_epochs = 5
    path_prefix = '/luckygong/data/train-10classes/all/'
    
    # init graph
    with slim.arg_scope(vgg16.vgg_arg_scope()):
        outputs, end_points = vgg16.vgg_16()
    
    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        for i in range(num_epochs):
            loader = Loader(input_file, delimiter, raw_size, processed_size, is_training, batch_size, num_prefetch,
                 num_threads, path_prefix, shuffle=True, inference_only=False)
            image_batch, label_batch, _ = loader.load()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess,coord)
            try:
                while not coord.should_stop():
                    i,l = sess.run([image_batch,label_batch])
                    print(str(i.shape) + '\t' + str(l.shape))
                    # sess.run(graph['optimizer'], feed_dict = {graph['x']:i,graph['y']:label_batch})
            except tf.errors.OutOfRangeError:
                print('Done training')
            finally:
                coord.request_stop()
                coord.join(threads)
                sess.close() 