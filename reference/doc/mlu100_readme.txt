一.加载驱动:
cd /home/cambricon/mlu100/Cambricon-Test/driver/MLU100_Driver_v2_0_3_CNCODEC_v1.0.1_x86_64
make
sudo ./load

三.进入docker环境
cd /home/cambricon/mlu100
./run-cambricon-test-docker.sh


四.配置环境变量
cd /home/Cambricon-Test
source env.sh

五.Caffe测试:
1.编译caffe源码 
cd /home/Cambricon-Test/caffe/src
rm -rf build
./build_caffe_mlu100_release.sh

2.安装库文件
cd /home/Cambricon-Test/caffe/tools
./copy.sh

3.caffe在线离线使用参数说明：
model_name   :alexnet、googlenet、inception-v3、resnet101、resnet152、resnet18、
              resnet34、 resnet50、squeezenet、vgg16、vgg19、mobilenet
sparsity     :dense、sparse
accuracy     :float16、fix8
batch_size   :单次处理的图片数量1、2、4
mode         :0：CPU模式，1：MLU（在线逐层），2：MFUS（在线融合）
threads      :线程数量（最多支持4个线程）
parallel     :每个线程使用的核数

4.online demo:
cd /home/Cambricon-Test/caffe/examples/classification/classification_online_multicore
make clean
make
./run.sh  model_name  sparsity  accuracy  mode batch_size threads  parallel
示例：
./run.sh alexnet dense float16 0 1 4 8

5.offline demo:
cd /home/Cambricon-Test/caffe/examples/classification/classification_offline_multicore
make clean
make
./run.sh model_name  sparsity  accuracy batch_size  threads parallel
示例：
./run.sh alexnet dense float16 1 4 8

6.ssd
cd /home/Cambricon-Test/caffe/examples/ssd/ssd_online_multicore
make clean
make
./run.sh  sparsity  accuracy  mode  batch_size  threads  parallel
示例：
./run.sh dense float16 0 1 4 8

cd /home/Cambricon-Test/caffe/examples/ssd/ssd_offline_multicore
make clean
make
./run.sh  sparsity  accuracy  batch_size  threads   parallel
示例：
./run.sh dense float16 1 4 8

六.cnml测试
cd  /home/Cambricon-Test/cnml/example/cnml_op_test
./make.sh
cd /home/Cambricon-Test/cnml/example/cnml_op_test/build
./cnmlTest


七.cnrt测试
cd /home/Cambricon-Test/cnrt/example
make clean
make
示例：
并行测试：./parallel
串行测试：./serial


八.tensorflow 测试
1.编译
cd /home/Cambricon-Test/tensorflow/tensorflow-v1.4
rm -rf virtualenv_mlu/
./build_tensorflow-v1.4_mlu.sh

2.生成离线模型
cd /home/Cambricon-Test/tensorflow/tools/pb_to_cambricon
编译：./build_host.sh
生成离线dense模型：./generate_cambricon_model_x86.sh
生成离线稀疏模型：./generate_cambricon_sparse_model_x86.sh

3.tensorflow在线离线使用参数说明：
  network                              [note]  resnet18_v1|resnet34_v1|resnet50_v2|resnet101_v2|resnet152_v2|
											   vgg16|vgg19|alexnet|squeezenet|inception_v1|inception_v3|mobileNet
  cpu/mlu                              [note]  Device to run
  dense/sparse/fix8_dense/fix8_sparse  [note]  chooose dense/sparse/fix8 model
  batch_size                           [note]  1/2/4/..
  number of image                      [note]  run number of image(32*batch_size*n)

3.online demo
cd /home/Cambricon-Test/tensorflow/examples/online/classification
./build_host.sh
示例：
./tensorflow-v1.4_online.sh resnet18_v1 mlu dense 1 32

4.offline demo
cd /home/Cambricon-Test/tensorflow/examples/offline/classification
./build_host.sh
示例：
./tensorflow-v1.4_offline.sh resnet18_v1 dense 32


九:mxnet
1.编译
cd /home/Cambricon-Test/mxnet/src
./build_mxnet_mlu_release.sh

2.mxnet使用参数说明：
model_name  :alexnet, googlenet, vgg16, vgg19, etc
mode        : dense sparse
datatype    : float16 or fix8
batch       : 1/2/4
output-mode : picture/screenonly/text(picture:plot picture, screenonly:print coordinate on screen, text:output to text)
gen_offlinemode: 0 or 1, 0:run online only, 1:generate offline model only

3.online demo
cd /home/Cambricon-Test/mxnet/examples/classification/classification_online
示例：
./run_all.sh alexnet 1

4.ssd demo
cd /home/Cambricon-Test/mxnet/examples/ssd/ssd_online
示例：
./run.sh dense float16 1 picture 0

