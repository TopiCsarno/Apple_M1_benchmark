testing

# BENCHMARK
- geekbench

# SETUP
LINUX SETUP:

  install cuda & cudnn proper version for tensorflow v2.x
  
  make sure versions match
  https://www.tensorflow.org/install/source#gpu

  different on each system
  https://developer.nvidia.com/cuda-toolkit-archive
  https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
  
  check cuda version and cudnn version (include/cudnn_version.h, nvcc -V)

  tested on: 
  cuda 11.1, cudnn 8.0.4

  
  ## Install dependencies (tested on this)
  requirements.txt

  new env
  pip install tensorflow (tf.test_is_gpu_available())
  pip install -v dlib -> log text should say DLIB WILL USE CUDA-> make sure uses CUDA (import dlib; dlib.DLIB_USE_CUDA)
  pip install face_recognition
  pip install torch
  pip install opencv-python

MAC SETUP:
  
  through anaconda (details needed)

COLAB SETUP:
  
  turn on GPU,
  download and run tests

# TESTS: 

  - what does each test script do
  
# SUMMARY

  - eredmények
  - jelenlegi helyzet

