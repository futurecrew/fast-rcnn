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

set step=%1
set model=%2
set imdb=%3
set log_file=experiments\logs\faster_rcnn\log_%year%%month%%day%_%hour%%min%%secs%.txt

if "%step%" equ "" (
  call :usage
  exit /b
)
if "%model%" equ "" (
  call :usage
  exit /b
)
if "%imdb%" equ "" (
  call :usage
  exit /b
)

run0.bat %step% %model% %imdb% | tee -a %log_file%


:usage
echo "run.bat step model imdb"
echo "e.g. run.bat 1 VGG_CNN_M_1024 voc_2007_trainval"
echo "e.g. run.bat 0 VGG16 imagenet_train"
