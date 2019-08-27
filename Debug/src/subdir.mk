################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/CVIPtexture.cu \
../src/a.cu \
../src/firstorder.cu \
../src/glcm.cu \
../src/glszm.cu \
../src/initialize.cu 

CC_SRCS += \
../src/cuda_op_kernel.cu.cc 

CC_DEPS += \
./src/cuda_op_kernel.cu.d 

OBJS += \
./src/CVIPtexture.o \
./src/a.o \
./src/cuda_op_kernel.cu.o \
./src/firstorder.o \
./src/glcm.o \
./src/glszm.o \
./src/initialize.o 

CU_DEPS += \
./src/CVIPtexture.d \
./src/a.d \
./src/firstorder.d \
./src/glcm.d \
./src/glszm.d \
./src/initialize.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


