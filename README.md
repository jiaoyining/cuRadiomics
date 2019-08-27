# cuRadiomics
Extracting Radiomic Features using CUDA for GPU-acceleration

## REQUIREMENTS

### CUDA
You must have an Nvidia Gpu on your system. 

### Python
Python 3.6 is used in my case.
If you use other version of python, you may need to do some slight grammer adjustments by yourself. 

### Tensorflow
Tensorflow 1.12.0 or higher version of tensorflow needs to be installed.

### OS
In my case, it is Ubuntu 16.04.

## Compile
The source code is packaged into a dynamic library.
You can directly use it or choose whether to compile it again by yourself.
If you want to change some parameters such as the binwidth of GLCM, you need to use the CMakeLists.txt file to compile it:

* cd
* cd curadiomics/build/
* cmake ..
* make

## Customize
If you want to change some parameters such as the binwidth of GLCM, 
1. you need to set the value of BIN_WIDTH in the "parameter.h" file;
2. you need to use the CMakeLists.txt file to recompile it to a .so file.

## Radiomics Feature Extraction

Since python is quite frequently used in machine learning and artificial intelligence, we call the cuRadiomics in python platform.
You can use the .yaml file to choose what kind of feature you want to extract. 
func_curadiomics.py contains the interface of feature extraction, 
in which you need to import the image, mask as numpy format 
(since in most cases, there would be a normalizarion process before feature extraction, and numpy array is mostly used as output format), 
and the address of .yaml.

Demo.py is an example of radiomics feature extration process.
