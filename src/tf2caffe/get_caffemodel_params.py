net = caffe.Net('/luckygong/data/VGG_ILSVRC_16_layers_deploy.prototxt', '/luckygong/data/vgg_16.caffemodel', caffe.TEST)

net_params = net.params.items()
# for k, v in net_params:
#     print(list(v[0].data))
#     break

with open('my2_weights', 'w') as f:
    for k, v in net_params:
        f.write(k + ' ' + str(list(v[0].data)) + '\n')