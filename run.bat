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
set log_file=experiments\logs\faster_rcnn\log_%year%%month%%day%_%hour%%min%%secs%.txt
tools\train_net.py --gpu 0 --solver models/VGG_CNN_M_1024/rpn/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --train_target rpn --cfg experiments/cfgs/faster_rcnn.yml --iters %iters% 2>&1 | tee %log_file%