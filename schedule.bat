timeout 1
tools\train_net.py --gpu 0 --solver models/VGG_CNN_M_1024/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --model_to_use frcnn --cfg experiments/cfgs/fast_rcnn_lazy.yml --iters 40000
tools\test_net.py --gpu 0 --def models/VGG_CNN_M_1024/test.prototxt --net output/fast_rcnn_lazy/voc_2007_trainval_with_ss/vgg_cnn_m_1024_fast_rcnn_with_ss_iter_40000.caffemodel --cfg experiments/cfgs/fast_rcnn.yml

cd E:\project\fast-rcnn-40c0b118204c1fc9267b5b99fd839df131ac870a
tools\train_net.py --gpu 0 --imdb voc_2007_trainval --solver models/VGG_CNN_M_1024/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --cfg experiments/cfgs/fast_rcnn.yml --iters 40000
tools\test_net.py --gpu 0 --def models/VGG_CNN_M_1024/test.prototxt --net output/fast_rcnn/voc_2007_trainval_with_ss/vgg_cnn_m_1024_fast_rcnn_with_ss_iter_40000.caffemodel --cfg experiments/cfgs/fast_rcnn.yml

