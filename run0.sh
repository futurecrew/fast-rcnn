#!/bin/bash

step=$1
model=$2
imdb=$3
gpu=$4

iters_rpn=80000
iters_frcnn=40000

step_1="false"
step_2="false"
step_3="false"
step_4="false"

if [ "$step" == "0" ]; then
  step_1="true"
  step_2="true"
  step_3="true"
  step_4="true"
fi
if [ "$step" == "1" ]; then
  step_1="true"
fi
if [ "$step" == "2" ]; then
  step_2="true"
fi
if [ "$step" == "3" ]; then
  step_3="true"
fi
if [ "$step" == "4" ]; then
  step_4="true"
fi


if [ "$model" == "VGG_CNN_M_1024" ]; then
  model_L=vgg_cnn_m_1024
fi
if [ "$model" == "VGG16" ]; then
  model_L=vgg16
fi
if [ "$model" == "GoogleNet" ]; then
  model_L=googlenet
fi

if [ "$step_1" == "true" ]; then
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 1 : Train RPN based on imagenet pre-trained model"
  echo "----------------------------------------------------------------------------------"
  cmd="tools/train_net.py --gpu $gpu --imdb $imdb --solver models/$model/rpn/solver.prototxt --weights data/imagenet_models/$model.v2.caffemodel --model_to_use rpn --cfg experiments/cfgs/faster_rcnn_lazy.yml --iters $iters_rpn"
  echo $cmd
  $cmd
  
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 1 : Generate RPN proposals"
  echo "----------------------------------------------------------------------------------"
  cmd="tools/find_candidate_object_locations_files.py --weights output/faster_rcnn_lazy/${imdb}/${model_L}_rpn_iter_${iters_rpn}.caffemodel --prototxt models/$model/rpn/test.prototxt --cfg experiments/cfgs/faster_rcnn_lazy.yml --data_type trainval --model_name ${model_L} --step 1"
  echo $cmd
  $cmd
  
  cmd="tools/find_candidate_object_locations_files.py --weights output/faster_rcnn_lazy/${imdb}/${model_L}_rpn_iter_${iters_rpn}.caffemodel --prototxt models/$model/rpn/test.prototxt --cfg experiments/cfgs/faster_rcnn_lazy.yml --data_type test --model_name ${model_L} --step 1"
  echo $cmd
  $cmd
fi

if [ "$step_2" == "true" ]; then
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 2 : Train FRCNN based on imagenet pre-trained model with step 1 RPN proposals"
  echo "----------------------------------------------------------------------------------"
  cmd="tools/train_net.py --gpu $gpu --imdb $imdb --solver models/$model/solver_step2.prototxt --weights data/imagenet_models/$model.v2.caffemodel --model_to_use frcnn --cfg experiments/cfgs/fast_rcnn_lazy.yml --proposal rpn --proposal_file=output/rpn_data/${imdb}/${model_L}_step_1_rpn_top_2300_candidate.pkl --iters $iters_frcnn"
  echo $cmd
  $cmd
fi

if [ "$step_3" == "true" ]; then
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 3 : Train RPN based on step 2 FRCNN with step 2 RPN proposals"
  echo "----------------------------------------------------------------------------------"
  cmd="tools/train_net.py --gpu $gpu --imdb $imdb --solver models/$model/rpn/solver_step3.prototxt --weights output/fast_rcnn_lazy/${imdb}_with_rpn/${model_L}_fast_rcnn_step2_with_rpn_iter_$iters_frcnn.caffemodel --model_to_use rpn --cfg experiments/cfgs/faster_rcnn_lazy.yml --iters $iters_rpn"
   echo $cmd
  $cmd
  
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 3 : Generate RPN proposals"
  echo "----------------------------------------------------------------------------------"
  cmd="tools/find_candidate_object_locations_files.py --weights output/faster_rcnn_lazy/${imdb}/${model_L}_rpn_step3_iter_$iters_rpn.caffemodel --prototxt models/$model/rpn/test.prototxt --cfg experiments/cfgs/faster_rcnn_lazy.yml --data_type trainval --model_name ${model_L} --step 3"
  echo $cmd
  $cmd

  cmd="tools/find_candidate_object_locations_files.py --weights output/faster_rcnn_lazy/${imdb}/${model_L}_rpn_step3_iter_$iters_rpn.caffemodel --prototxt models/$model/rpn/test.prototxt --cfg experiments/cfgs/faster_rcnn_lazy.yml --data_type test --model_name ${model_L} --step 3"
  echo $cmd
  $cmd

  cmd="tools/find_candidate_object_locations_files.py --weights output/faster_rcnn_lazy/${imdb}/${model_L}_rpn_step3_iter_$iters_rpn.caffemodel --prototxt models/$model/rpn/test.prototxt --cfg experiments/cfgs/faster_rcnn_lazy.yml --data_type test --model_name ${model_L} --step 3  --max_output 300"
  echo $cmd
  $cmd
fi

if [ "$step_4" == "true" ]; then
  echo ""
  echo "----------------------------------------------------------------------------------"
  echo "Step 4 : Train FRCNN based on step 2 FRCNN with step 3 RPN proposals"
  echo "----------------------------------------------------------------------------------"
  cmd="tools/train_net.py --gpu $gpu --imdb $imdb --solver models/$model/solver_step4.prototxt --weights output/fast_rcnn_lazy/${imdb}_with_rpn/${model_L}_fast_rcnn_step2_with_rpn_iter_$iters_frcnn.caffemodel --model_to_use frcnn --cfg experiments/cfgs/fast_rcnn_lazy.yml --proposal rpn --proposal_file=output/rpn_data/${imdb}/${model_L}_step_3_rpn_top_2300_candidate.pkl --iters $iters_frcnn"
  echo $cmd
  $cmd
fi

