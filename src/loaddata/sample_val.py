# -*- coding:utf-8 -*-
import shutil

with open('../../data/val-dic.txt') as dic_f:
    for line in dic_f:
        shutil.copyfile('../../data/ImageNet/eval/' + line.strip().split('\t')[0], '/home/ictwsn/luckygong/data/ImageNet/eval-10classes/' + line.strip().split('\t')[0])
