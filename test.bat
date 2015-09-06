@echo off
set step=%1

echo "step=%step%"

if "%step%" equ "" (
  echo "HAHA"
) else (
  if "%step%" equ "1" (
    echo "HAHA1"
  )
  if "%step%" equ "2" (
    echo "HAHA2"
  )
)

ping localhost | tee haha.txt