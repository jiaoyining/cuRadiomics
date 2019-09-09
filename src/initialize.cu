#include"initialize.h"

__global__ void initialize(int *glcm)
{
    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip = blocks * blockDim.x * blockDim.y + threads;

    glcm[ip] = 0;

}
__global__ void initialize_tex(float *texture){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	atomicExch(&texture[ip], 0.0);
	//printf("%f", texture_glcm[0]);

}

__global__ void initialize_mtex(float *texture){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	atomicExch(&texture[ip], 255.0);
	//printf("%f", texture_glcm[0]);

}


__global__ void Preprocessing_image_GLCM(int *dev_image, const int *image, int Min_V, int Max_V, int bin_width, int MASK_V){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	atomicExch(&dev_image[ip], image[ip]==-1?-1:(image[ip]-Min_V)/bin_width);

}

__global__ void Preprocessing_image_firstorder(int *dev_image, const int *image, int Min_V, int Max_V, int MASK_V){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	atomicExch(&dev_image[ip], image[ip]-Min_V);

}


