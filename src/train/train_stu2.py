# coding=utf-8

import sys
sys.path.append('../model')
sys.path.append('../loaddata')
import vgg16
import tensorflow as tf
import time

from data_loader import Loader

slim = tf.contrib.slim

if __name__ == '__main__':

	# load data
    train_input_file = '/luckygong/data/train-dic3.txt'
    eval_input_file = '/luckygong/data/val-dic.txt'
    delimiter = '\t'
    raw_size = [224, 224, 3]
    processed_size = [224, 224, 3]
    num_classes = 10
    is_training = True
    batch_size = 64
    all_eval_data_size = 500
    eval_batch_size = 50
    num_prefetch = batch_size
    eval_num_prefetch = eval_batch_size
    num_threads = 1
    num_epochs = 5
    train_path_prefix = '/luckygong/data/train-10classes/all/'
    eval_path_prefix = '/luckygong/data/eval-10classes/all/'
    start_time = time.time()

    # init graph
    with slim.arg_scope(vgg16.vgg_arg_scope()):
        outputs, end_points = vgg16.vgg_16()
    
    # train
    with tf.Session() as sess:
        train_loader = Loader(train_input_file, delimiter, raw_size, processed_size, is_training, batch_size, num_prefetch,
                 num_threads, train_path_prefix, num_classes, num_epochs, shuffle=True, inference_only=False)
        train_image_batch, train_label_batch, _ = train_loader.load()

        eval_loader = Loader(eval_input_file, delimiter, raw_size, processed_size, is_training, eval_batch_size, eval_num_prefetch,
                 num_threads, eval_path_prefix, num_classes, num_epochs, shuffle=True, inference_only=False)
        eval_image_batch, eval_label_batch, _ = eval_loader.load()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        for j in range(num_epochs):
            try:
                now_step = 1
                while not coord.should_stop():
                    i,l = sess.run([train_image_batch, train_label_batch])
                    # print(i[0][0][0])
                    _, loss, acc = sess.run([end_points['optimizer'], end_points['loss'], end_points['accuracy']], feed_dict = {end_points['x']:i, end_points['y']:l})
                    print('run time: ' + str(time.time() - start_time) + '\tbatch: ' + str(j + 1)  + '\tstep: '  + str(now_step) + '\tdata size: ' + str(i.shape) + ', ' + str(l.shape) + '\tbatch loss: ' + str(loss) + '\tacc: ' + str(acc))
                    if(now_step % 100 == 0):
                        correct_times = 0
                        all_times = 0
                        all_eval_step = int(all_eval_data_size / eval_batch_size)
                        if(all_eval_data_size % eval_batch_size != 0):
                            all_eval_step += 1
                        for k in range(all_eval_step):
                            e_i, e_l = sess.run([eval_image_batch, eval_label_batch])
                            correct_times += sess.run(end_points['correct_times'], feed_dict = {end_points['x']: e_i, end_points['y']: e_l})
                            all_times += len(e_i)
                        print('Eval: all eval data size: ' + str(all_times) + ', ' + '\tcorrect eval data size: ' + str(correct_times) + '\tacc: ' + str(correct_times * 1.0 / all_times))
                    now_step += 1
            except tf.errors.OutOfRangeError:
                print('Done training')
            finally:
                coord.request_stop()
                coord.join(threads)
                sess.close() 