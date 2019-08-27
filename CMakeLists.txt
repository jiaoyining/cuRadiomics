    CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
    PROJECT(Radiomics)
    find_package(CUDA REQUIRED)
    SET(CMAKE_CUDA_COMPILER  /usr/local/cuda-9.2/bin/nvcc)
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} 
	-O2
	-gencode arch=compute_61,code=sm_61 
	-D GOOGLE_CUDA=1 
	-x cu 
	-Xcompiler
	-fPIC
	-D_GLIBCXX_USE_CXX11_ABI=0")
    EXECUTE_PROCESS(COMMAND python3.6 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_include(), end='', flush=True)"  OUTPUT_VARIABLE TF_INC)

EXECUTE_PROCESS(COMMAND python3.6 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_lib(), end='', flush=True)"  OUTPUT_VARIABLE TF_LIB)

    SET(TENSORFLOW_INCLUDE_PATH /usr/local/include/google)
    SET(TENSORFLOW_CC_LIBARY /usr/local/lib/libtensorflow_cc.so)
    SET(TENSORFLOW_LIBARY /usr/local/lib/libtensorflow.so)
    SET(TENSORFLOW_FRAMEWORK_LIBARY /usr/local/lib/libtensorflow_framework.so)
    MESSAGE(STATUS "TENSORFLOW_INCLUDE_PATH ${TENSORFLOW_INCLUDE_PATH}")
    MESSAGE(STATUS "TENSORFLOW_LIBARY ${TENSORFLOW_LIBARY}")
    MESSAGE(STATUS "TENSORFLOW_FRAMEWORK_LIBARY ${TENSORFLOW_FRAMEWORK_LIBARY}")
    SET(EIGEN_INCLUDE_PATH /usr/local/include/eigen3)
    SET(ABSL_INCLUDE_PATH /usr/local/include/google/absl)
    MESSAGE(STATUS "ABSL_INCLUDE_PATH" ${ABSL_INCLUDE_PATH})
    INCLUDE(FindProtobuf)

    FIND_PACKAGE(Protobuf REQUIRED)
    INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(${TENSORFLOW_INCLUDE_PATH})
    INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_PATH})
    INCLUDE_DIRECTORIES(${ABSL_INCLUDE_PATH})
    SET(EXAMPLE_LIBRARIES
        ${PROTOBUF_LIBRARY}
        ${TENSORFLOW_FRAMEWORK_LIBARY}
        #${TENSORFLOW_LIBARY}
	${TENSORFLOW_CC_LIBARY}
	)
    SET(LDFLAGS "-std=c++11 -ltensorflow_framework -ltensorflow_cc")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}${LDFLAGS}")
    MESSAGE(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
    MESSAGE(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    SET(SRC_PATH "/home/jyn/cuRadiomics/curadiomics/src/")

#    CUDA_ADD_LIBRARY(UTILS_CU_O SHARED 
#	            ./src/utils.h 
#		     ./src/helper_string.h 
#                     ./src/helper_cuda.h 
#                     ./src/utils.cu 
#                     MODULE OPTIONS -I$TF_INC -I/usr/local/include/google/)
#    CUDA_ADD_LIBRARY(CVIPTEXTURE_CU_O SHARED
#                     ./src/CVIPtexture.h
#                     ./src/CVIPtexture.cu 
#                      MODULE OPTIONS -I$TF_INC -I/usr/local/include/google/)
    
#CUDA_ADD_LIBRARY(UTILS_CU_O SHARED
#	     	     ./src/helper_string.h
#		     ./src/helper_cuda.h
#		     ./src/utils.h
#		     ./src/utils.cu
#		    )
#   TARGET_LINK_LIBRARIES(UTILS_CU_O ${EXAMPLE_LIBRARIES} ${CMAKE_CXX_FLAGS})
    #CUDA_ADD_SUBDIRECTORY(UTILS_CU_O)
    #CUDA_ADD_SUBDIRECTORY(UTILS_CU_O)
    CUDA_ADD_LIBRARY(Radiomics SHARED
#		     ${UTILS_CU_O}
#		     ${CVIPTEXTURE_CU_O}
    		     src/cuda_op_kernel.cu.cc
		     src/CVIPtexture.h
		     src/CVIPtexture.cu
		     src/initialize.h
		     src/initialize.cu
		     src/firstorder.h
	 	     src/firstorder.cu
		     src/glcm.h
		     src/glcm.cu
		     src/headers.h
		    )
    TARGET_LINK_LIBRARIES(Radiomics  ${EXAMPLE_LIBRARIES} ${CMAKE_CXX_FLAGS})
