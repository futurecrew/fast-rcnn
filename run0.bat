set iters_rpn=80000
set iters_frcnn=40000

set step_1=false
set step_2=false
set step_3=false
set step_4=false

if "%step%" equ "0" (
  set step_1=true
  set step_2=true
  set step_3=true
  set step_4=true
) else (
  if "%step%" equ "1" set step_1=true
  if "%step%" equ "2" set step_2=true
  if "%step%" equ "3" set step_3=true
  if "%step%" equ "4" set step_4=true
)


if "%model%" equ "VGG_CNN_M_1024" (
  set model_L=vgg_cnn_m_1024
)
if "%model%" equ "VGG16" (
  set model_L=vgg16
)
if "%model%" equ "GoogleNet" (
  set model_L=googlenet
)

if "%step_1%" equ "true" (
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 1 : Train RPN based on imagenet pre-trained model"
  echo "----------------------------------------------------------------------------------"
  tools\train_net.py --gpu 0 --imdb %imdb% --solver models/%model%/rpn/solver.prototxt --weights data/imagenet_models/%model%.v2.caffemodel --model_to_use rpn --cfg experiments/cfgs/faster_rcnn_lazy.yml --iters %iters_rpn% 2>&1
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 1 : Generate RPN proposals"
  echo "----------------------------------------------------------------------------------"
  tools\find_candidate_object_locations_files.py --weights output/faster_rcnn_lazy/voc_2007_trainval/%model_L%_rpn_iter_%iters_rpn%.caffemodel --prototxt models/%model%/rpn/test.prototxt --cfg experiments/cfgs/faster_rcnn_lazy.yml --data_type trainval --model_name %model_L% --step 1
  tools\find_candidate_object_locations_files.py --weights output/faster_rcnn_lazy/voc_2007_trainval/%model_L%_rpn_iter_%iters_rpn%.caffemodel --prototxt models/%model%/rpn/test.prototxt --cfg experiments/cfgs/faster_rcnn_lazy.yml --data_type test --model_name %model_L% --step 1
)

if "%step_2%" equ "true" (
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 2 : Train FRCNN based on imagenet pre-trained model with step 1 RPN proposals"
  echo "----------------------------------------------------------------------------------"
  tools\train_net.py --gpu 0 --imdb %imdb% --solver models/%model%/solver_step2.prototxt --weights data/imagenet_models/%model%.v2.caffemodel --model_to_use frcnn --cfg experiments/cfgs/fast_rcnn_lazy.yml --proposal rpn --proposal_file=data/rpn_data/voc_2007_trainval_step_1_rpn_top_2300_candidate.pkl --iters %iters_frcnn% 2>&1
)

if "%step_3%" equ "true" (
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 3 : Train RPN based on step 2 FRCNN with step 2 RPN proposals"
  echo "----------------------------------------------------------------------------------"
  tools\train_net.py --gpu 0 --imdb %imdb% --solver models/%model%/rpn/solver_step3.prototxt --weights output/fast_rcnn_lazy/voc_2007_trainval_with_rpn/%model_L%_fast_rcnn_step2_with_rpn_iter_%iters_frcnn%.caffemodel --model_to_use rpn --cfg experiments/cfgs/faster_rcnn_lazy.yml --iters %iters_rpn% 2>&1
  
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 3 : Generate RPN proposals"
  echo "----------------------------------------------------------------------------------"
  tools\find_candidate_object_locations_files.py --weights output/faster_rcnn_lazy/voc_2007_trainval/%model_L%_rpn_step3_iter_%iters_rpn%.caffemodel --prototxt models/%model%/rpn/test.prototxt --cfg experiments/cfgs/faster_rcnn_lazy.yml --data_type trainval --model_name %model_L% --step 3
  tools\find_candidate_object_locations_files.py --weights output/faster_rcnn_lazy/voc_2007_trainval/%model_L%_rpn_step3_iter_%iters_rpn%.caffemodel --prototxt models/%model%/rpn/test.prototxt --cfg experiments/cfgs/faster_rcnn_lazy.yml --data_type test --model_name %model_L% --step 3
  tools\find_candidate_object_locations_files.py --weights output/faster_rcnn_lazy/voc_2007_trainval/%model_L%_rpn_step3_iter_%iters_rpn%.caffemodel --prototxt models/%model%/rpn/test.prototxt --cfg experiments/cfgs/faster_rcnn_lazy.yml --data_type test --model_name %model_L% --step 3  --max_output 300
)

if "%step_4%" equ "true" (
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 4 : Train FRCNN based on step 2 FRCNN with step 3 RPN proposals"
  echo "----------------------------------------------------------------------------------"
  tools\train_net.py --gpu 0 --imdb %imdb% --solver models/%model%/solver_step4.prototxt --weights output/fast_rcnn_lazy/voc_2007_trainval_with_rpn/%model_L%_fast_rcnn_step2_with_rpn_iter_%iters_frcnn%.caffemodel --model_to_use frcnn --cfg experiments/cfgs/fast_rcnn_lazy.yml --proposal rpn --proposal_file=data/rpn_data/voc_2007_trainval_step_3_rpn_top_2300_candidate.pkl --iters %iters_frcnn% 2>&1
)
