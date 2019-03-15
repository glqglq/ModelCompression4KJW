#! /usr/bin/env python3
python re_checkpoint.py \
--ckpt_path /code/slim/param \
--save_path /code/slim/param_re_out/model.ckpt \
--rename_var_src conv1_1,conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,conv3_3,conv4_1,conv4_2,conv4_3,conv5_1,conv5_2,conv5_3,fc6,fc7,fc8 \
--rename_var_dst vgg_16/conv1/conv1_1,vgg_16/conv1/conv1_2,vgg_16/conv2/conv2_1,vgg_16/conv2/conv2_2,vgg_16/conv3/conv3_1,vgg_16/conv3/conv3_2,vgg_16/conv3/conv3_3,vgg_16/conv4/conv4_1,vgg_16/conv4/conv4_2,vgg_16/conv4/conv4_3,vgg_16/conv5/conv5_1,vgg_16/conv5/conv5_2,vgg_16/conv5/conv5_3,vgg_16/fc6,vgg_16/fc7,vgg_16/fc8
