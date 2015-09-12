timeout 7200
tools\test_net.py --gpu 0 --def models/Googlenet/test.prototxt --net output/fast_rcnn/voc_2007_trainval_with_ss/googlenet_1_4_fast_rcnn_with_ss_iter_100000.caffemodel --cfg experiments/cfgs/fast_rcnn.yml
