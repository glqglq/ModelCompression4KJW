import numpy as np
import tensorflow as tf

from numpy import *
test=np.load('/tmp/tfmodel_pre_conv/vgg16.npy',encoding = "latin1").item()  #加载文件
doc = open('/tmp/tfmodel_pre_conv/1.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)  #将打印内容写入文件中


# for item in test:
#     print(test[item])
#
#
# print('done')
#print(test['fc8'])

import numpy as np

data_dict = np.load('/tmp/tfmodel_pre_conv/vgg16.npy', encoding='latin1').item()
keys = sorted(data_dict.keys())
for key in keys:
    weights = data_dict[key][0]
    biases = data_dict[key][1]
    print('\n')
    print(key)
    print('weights shape: ', weights.shape)
    print('biases shape: ', biases.shape)
