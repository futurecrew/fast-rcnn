@echo off

set hour=%time:~0,2%
if "%hour:~0,1%" == " " set hour=0%hour:~1,1%
set min=%time:~3,2%
if "%min:~0,1%" == " " set min=0%min:~1,1%
set secs=%time:~6,2%
if "%secs:~0,1%" == " " set secs=0%secs:~1,1%

set year=%date:~0,4%
set month=%date:~5,2%
if "%month:~0,1%" == " " set month=0%month:~1,1%
set day=%date:~8,2%
if "%day:~0,1%" == " " set day=0%day:~1,1%

set iters=%1
set step=%2
set log_file=experiments\logs\faster_rcnn\log_%year%%month%%day%_%hour%%min%%secs%.txt

if %step% equ 1 (
  echo "Step 1 : Train RPN"
  tools\train_net.py --gpu 0 --solver models/VGG_CNN_M_1024/rpn/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --model_to_use rpn --cfg experiments/cfgs/faster_rcnn.yml --iters %iters% 2>&1 | tee %log_file%
  
  echo "Step 1 : Generate RPN proposals"
  tools\find_candidate_object_locations_files.py --weights output/faster_rcnn/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_%iters%.caffemodel --cfg experiments/cfgs/faster_rcnn.yml --data_type trainval --step 1
  tools\find_candidate_object_locations_files.py --weights output/faster_rcnn/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_%iters%.caffemodel --cfg experiments/cfgs/faster_rcnn.yml --data_type test --step 1  
)

if %step% equ 2 (
  echo "Step 2 : Train FRCNN with step 1 RPN proposals"
  tools\train_net.py --gpu 0 --solver models/VGG_CNN_M_1024/solver_step2.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --model_to_use frcnn --cfg experiments/cfgs/fast_rcnn.yml --proposal rpn --proposal_file=data/rpn_data/voc_2007_trainval_step_1_rpn_top_2300_candidate.pkl --iters %iters% 2>&1 | tee %log_file%
)
if %step% equ 3 (
  echo "Step 3 : Train RPN with step 2 FRCNN"
  tools\train_net.py --gpu 0 --solver models/VGG_CNN_M_1024/rpn/solver_step3.prototxt --weights output/fast_rcnn/voc_2007_trainval_with_rpn/vgg_cnn_m_1024_fast_rcnn_with_rpn_iter_%iters%.caffemodel --model_to_use rpn --cfg experiments/cfgs/faster_rcnn.yml --iters %iters% 2>&1 | tee %log_file%
  
  echo "Step 3 : Generate RPN proposals"
  tools\find_candidate_object_locations_files.py --weights output/faster_rcnn/voc_2007_trainval/vgg_cnn_m_1024_rpn_step3_iter_%iters%.caffemodel --cfg experiments/cfgs/faster_rcnn.yml --data_type trainval --step 3
  tools\find_candidate_object_locations_files.py --weights output/faster_rcnn/voc_2007_trainval/vgg_cnn_m_1024_rpn_step3_iter_%iters%.caffemodel --cfg experiments/cfgs/faster_rcnn.yml --data_type test --step 3  --max_output 300
)
if %step% equ 4 (
  echo "Step 4 : Train FRCNN with step 3 RPN proposals"
  tools\train_net.py --gpu 0 --solver models/VGG_CNN_M_1024/solver_step4.prototxt --weights output/fast_rcnn/voc_2007_trainval_with_rpn/vgg_cnn_m_1024_fast_rcnn_with_rpn_iter_%iters%.caffemodel --model_to_use frcnn --cfg experiments/cfgs/fast_rcnn.yml --proposal rpn --proposal_file=data/rpn_data/voc_2007_trainval_step_3_rpn_top_2300_candidate.pkl --iters %iters% 2>&1 | tee %log_file%
)
