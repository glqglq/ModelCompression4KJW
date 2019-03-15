# import tensorflow as tf
#
# sess = tf.Session()
# check_point_path = '/code/slim/param_re_out'
# saver = tf.train.import_meta_graph('/code/slim/param_re_out/param_re_out.meta')
#
# saver.restore(sess, tf.train.latest_checkpoint(check_point_path))
#
# graph = tf.get_default_graph()
#
# # print(graph.get_operations())
#
# # with open('op.txt','a') as f:
# #    f.write(str(graph.get_operations()))
# # op1 = graph.get_tensor_by_name('conv1_1/biases:0')
# # print(op1)
# # print(graph)
#

import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
check_point_path = '/docker_data/ModelCompression/model/student_hinton'
#checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
checkpoint_path = os.path.join('.', ckpt.model_checkpoint_path)
#print(ckpt.model_checkpoint_path)
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(reader.get_tensor(key))

