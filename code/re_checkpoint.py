# -*- coding:utf-8 -*-
#!/usr/bin/env python

'''
############################################################
rename tensorflow variable.
############################################################
'''

import tensorflow as tf
import argparse
import os
import re

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to rename tensorflow variable!')
    parser.add_argument('--ckpt_path', type=str, help='the ckpt file where to load.')
    parser.add_argument('--save_path', type=str, help='the ckpt file where to save.')
    parser.add_argument('--rename_var_src', type=str, help="""Comma separated list of replace variable from""")
    parser.add_argument('--rename_var_dst', type=str, help="""Comma separated list of replace variable to""")
    parser.add_argument('--add_prefix', type=str, help='prefix of newname.')
    args = parser.parse_args()
    return args

def load_model(model_path, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model_path)
    if (os.path.isfile(model_exp)):
        print('not support: %s' % model_exp)
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    return saver

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def rename(args):
    '''rename tensorflow variable, just for checkpoint file format.'''

    replace_from = args.rename_var_src.strip().split(',')
    #print('replace_from is ',replace_from)
    replace_to = args.rename_var_dst.strip().split(',')
    #print('replace_to is ', replace_to)
    assert len(replace_from) == len(replace_to)

    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(args.ckpt_path):
            #Returns list of all variables in the latest checkpoint.

            # Load the variable
            var = tf.contrib.framework.load_variable(args.ckpt_path, var_name)
            # 返回具有检查点中给定变量内容的Tensor。

            # Set the new name
            new_name = var_name

            for index in range(len(replace_from)):
                #print('replace_from is ',replace_from)
                #print('replace_to is ',replace_to)
                new_name = new_name.replace(replace_from[index], replace_to[index])
                #new_name[index] = replace_to[index]
                print('new_name is ',new_name)

            if args.add_prefix:
                new_name = args.add_prefix + new_name

            print('Renaming %s to %s.' % (var_name, new_name))
            # Rename the variable
            var = tf.Variable(var, name=new_name)
            print('name is ',var.name)
            print('var is : ',var)

        # Save the variables

        saver = load_model(args.ckpt_path)
        sess.run(tf.global_variables_initializer())
        print(args.save_path,'vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        saver.save(sess, args.save_path)

if __name__ == '__main__':
    args = get_parser()
    print('bbbbbbbbbbbbbbbbbbbbbbbbbbb')
    print(args,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    rename(args)