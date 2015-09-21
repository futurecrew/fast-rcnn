#!/bin/bash

step=$1
model=$2
imdb=$3
gpu=$4

function usage
{
echo "run.bat step model imdb gpu_no"
echo "e.g. run.bat 1 VGG_CNN_M_1024 voc_2007_trainval 0"
echo "e.g. run.bat 0 VGG16 imagenet_train 0"
}

cur_time=`date +%Y%m%d_%H%M%S`

log_file=experiments/logs/faster_rcnn/log_${cur_time}.txt

echo "log_file : ${log_file}"

if [ "$step" == "" ]; then
  usage
  exit 
fi
if [ "$model" == "" ]; then
  usage
  exit 
fi
if [ "$imdb" == "" ]; then
  usage
  exit 
fi
if [ "$gpu" == "" ]; then
  usage
  exit 
fi

bash run0.sh ${step} ${model} ${imdb} ${gpu} 2>&1 | tee -a ${log_file}


