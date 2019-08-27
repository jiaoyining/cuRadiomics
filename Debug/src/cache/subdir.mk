################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cache/demo.cu \
../src/cache/glcm_gpu.cu \
../src/cache/utils.cu 

OBJS += \
./src/cache/demo.o \
./src/cache/glcm_gpu.o \
./src/cache/utils.o 

CU_DEPS += \
./src/cache/demo.d \
./src/cache/glcm_gpu.d \
./src/cache/utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/cache/%.o: ../src/cache/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "src/cache" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


