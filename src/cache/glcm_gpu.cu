/*
include "glcm_gpu.h"
#include "./helper/utils.h"
#include "glcm_cpu.h"


#define NSIZE 3
#define NSTRIDE 3
#define NANGLES 42

__constant__ int dev_size[NSIZE];
__constant__ int dev_stride[NSTRIDE];
__constant__ int dev_ng = 31;
__constant__ int dev_na = 13;

int size[] = {155, 240, 240};
int stride[] = {57600, 240, 1};
int na = 13, ng = 31;

int angles[] = {
    1, 1, 1,
    1, 1, 0,
    1, 1, -1,
    1, 0, 1,
    1, 0, 0,
    1, 0, -1,
    1, -1, 1,
    1, -1, 0,
    1, -1, -1,
    0, 1, 1,
    0, 1, 0,
    0, 1, -1,
    0, 0, 1,
    0, 0, 0};


__global__ void kernel(int *image, int *mask, int *threads_glcm, int *dev_angles)
{
    int iy = blockIdx.x;
    int ix = threadIdx.x;

    int idx = iy * blockDim.x + ix;

    int iz = 0;
    int a, j, ipix, glcm_idx;
    int offset = 0;
    for (iz = 0; iz < dev_size[0]; iz++)
    {
       ipix = iz * dev_stride[0] + iy * dev_stride[1] + ix;
        if (mask[ipix])
        {

            for (a = 0; a < dev_na; a++)
            {
                if (iz + dev_angles[a * 3] >= 0 && iz + dev_angles[a * 3] < dev_size[0] &&
                    iy + dev_angles[a * 3 + 1] >= 0 && iy + dev_angles[a * 3 + 1] < dev_size[1] &&
                    ix + dev_angles[a * 3 + 2] >= 0 && ix + dev_angles[a * 3 + 2] < dev_size[2])
                {
                    j = ipix + dev_angles[a * 3] * dev_stride[0] +
                        dev_angles[a * 3 + 1] * dev_stride[1] +
                        dev_angles[a * 3 + 2] * dev_stride[2];
                    if (mask[j])
                    {
                        glcm_idx = a + (image[j] - 1) * dev_na + (image[ipix] - 1) * dev_na * dev_ng;
                        offset = idx * dev_ng * dev_ng * dev_na;
                        threads_glcm[offset + glcm_idx]++;
                    }
                }
            }
        }
    }
}

__global__ void reduce_kernel(int *threads_glcm, int *glcm)
{

    int glcm_idx = blockDim.x * blockIdx.x + threadIdx.x;

    int i = 0, offset = 0;
    for (i = 0; i < dev_size[1] * dev_size[2]; i++)
    {
        offset = i * dev_ng * dev_ng * dev_na;
        glcm[glcm_idx] += threads_glcm[offset + glcm_idx];
    }
}

//use atomicAdd() of cuda
__global__ void calculate_glcm_kernel(int *image, int *mask, int *glcm, int *dev_angles)
{
    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ipix = blocks * blockDim.x * blockDim.y + threads;

    int j, glcm_idx, a, iz, iy, ix;

    if (mask[ipix])
    {
        iz = (ipix / dev_stride[0]);
        iy = (ipix % dev_stride[0]) / dev_stride[1];
        ix = (ipix % dev_stride[0]) % dev_stride[1];

        for (a = 0; a < dev_na; a++)
        {
            if (iz + dev_angles[a * 3] >= 0 && iz + dev_angles[a * 3] < dev_size[0] &&
                iy + dev_angles[a * 3 + 1] >= 0 && iy + dev_angles[a * 3 + 1] < dev_size[1] &&
                ix + dev_angles[a * 3 + 2] >= 0 && ix + dev_angles[a * 3 + 2] < dev_size[2])
            {
                j = ipix + dev_angles[a * 3] * dev_stride[0] +
                    dev_angles[a * 3 + 1] * dev_stride[1] +
                    dev_angles[a * 3 + 2] * dev_stride[2];
                if (mask[j])
                {
                    glcm_idx = a + (image[j] - 1) * dev_na + (image[ipix] - 1) * dev_na * dev_ng;
                    atomicAdd(&glcm[glcm_idx], 1);
                }
            }
        }
    }
}

void atomic_glcm_gpu(int *image, int *mask, int *glcm)
{

    int *dev_image;
    int *dev_mask;
    int *dev_glcm;
    int *dev_angles;

    // malloc gpu memory
    HANDLE_ERROR(cudaMalloc((void **)&dev_image, sizeof(int) * size[0] * size[1] * size[2]));
    HANDLE_ERROR(cudaMalloc((void **)&dev_mask, sizeof(int) * size[0] * size[1] * size[2]));
    HANDLE_ERROR(cudaMalloc((void **)&dev_glcm, sizeof(int) * ng * ng * na));
    HANDLE_ERROR(cudaMalloc((void **)&dev_angles, sizeof(int) * NANGLES));

    // copy to gpu
    HANDLE_ERROR(cudaMemcpy(dev_image, image, sizeof(int) * size[0] * size[1] * size[2], cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_mask, mask, sizeof(int) * size[0] * size[1] * size[2], cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_size, size, sizeof(int) * NSIZE));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_stride, stride, sizeof(int) * NSTRIDE));
    HANDLE_ERROR(cudaMemcpy(dev_angles, angles, sizeof(int) * NANGLES, cudaMemcpyHostToDevice));

    dim3 grids(155, 1, 1);
    dim3 threads(240, 240);

    START_TIMER(atomic_kernel_time)
    calculate_glcm_kernel<<<grids, threads>>>(dev_image, dev_mask, dev_glcm, dev_angles);
    HANDLE_ERROR(cudaDeviceSynchronize());
    STOP_TIMER(atomic_kernel_time)
    PRINT_TIME(atomic_kernel_time)

    HANDLE_ERROR(cudaMemcpy(glcm, dev_glcm, sizeof(int) * ng * ng * na, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_image));
    HANDLE_ERROR(cudaFree(dev_mask));
    HANDLE_ERROR(cudaFree(dev_glcm));
    HANDLE_ERROR(cudaFree(dev_angles));
}

void reduce_glcm_gpu(int *image, int *mask, int *glcm)
{

    int *dev_image;
    int *dev_mask;
    int *dev_glcm;
    int *dev_thread_glcm;
    int *dev_angles;

    // malloc gpu memory
    HANDLE_ERROR(cudaMalloc((void **)&dev_image, sizeof(int) * size[0] * size[1] * size[2]));
    HANDLE_ERROR(cudaMalloc((void **)&dev_mask, sizeof(int) * size[0] * size[1] * size[2]));

    HANDLE_ERROR(cudaMalloc((void **)&dev_glcm, sizeof(int) * ng * ng * na));
    HANDLE_ERROR(cudaMalloc((void **)&dev_angles, sizeof(int) * NANGLES));

    HANDLE_ERROR(cudaMalloc((void **)&dev_thread_glcm, sizeof(int) * ng * ng * na * size[1] * size[2]));

    // copy to gpu
    HANDLE_ERROR(cudaMemcpy(dev_image, image, sizeof(int) * size[0] * size[1] * size[2], cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_mask, mask, sizeof(int) * size[0] * size[1] * size[2], cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpyToSymbol(dev_size, size, sizeof(int) * NSIZE));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_stride, stride, sizeof(int) * NSTRIDE));
    HANDLE_ERROR(cudaMemcpy(dev_angles, angles, sizeof(int) * NANGLES, cudaMemcpyHostToDevice));

    dim3 grids(size[2]);
    dim3 threads(size[1]);
    dim3 grids_reduce(ng);
    dim3 threads_reduce(ng * na);

    START_TIMER(reduce_kernel_time)
    kernel<<<grids, threads>>>(dev_image, dev_mask, dev_thread_glcm, dev_angles);
    reduce_kernel<<<grids_reduce, threads_reduce>>>(dev_thread_glcm, dev_glcm);
    HANDLE_ERROR(cudaDeviceSynchronize());
    STOP_TIMER(reduce_kernel_time)
    PRINT_TIME(reduce_kernel_time)

    HANDLE_ERROR(cudaMemcpy(glcm, dev_glcm, sizeof(int) * ng * ng * na, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_image));
    HANDLE_ERROR(cudaFree(dev_mask));
    HANDLE_ERROR(cudaFree(dev_thread_glcm));
    HANDLE_ERROR(cudaFree(dev_angles));
}

void glcm_cpu(int *image, int *mask, int *glcm)
{

    START_TIMER(glcm_cpu_time)
    calculate_glcm(image, mask, size, stride, angles, na, glcm, ng);
    STOP_TIMER(glcm_cpu_time)
    PRINT_TIME(glcm_cpu_time)
}

void glcm_cpu_parallel(int *image,int*mask,int *glcm){
    START_TIMER(glcm_parallel_cpu_time)
    calculate_glcm_parallel(image, mask, size, stride, angles, na, glcm, ng);
    STOP_TIMER(glcm_parallel_cpu_time)
    PRINT_TIME(glcm_parallel_cpu_time)
}

int main()
{
	int dev = 0;
	    cudaDeviceProp devProp;
	    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
	    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
	    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
	    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
	    std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;


    int *image;
    int *mask;
    int *glcm, *reduce_glcm, *atomic_glcm,*para_glcm;

    image = (int *)malloc(sizeof(int) * size[0] * size[1] * size[2]);
    mask = (int *)malloc(sizeof(int) * size[0] * size[1] * size[2]);
    glcm = (int *)malloc(sizeof(int) * ng * ng * na);
    para_glcm = (int *)malloc(sizeof(int) * ng * ng * na);
    reduce_glcm = (int *)malloc(sizeof(int) * ng * ng * na);
    atomic_glcm = (int *)malloc(sizeof(int) * ng * ng * na);

    load_data(image, mask, size);
    glcm_cpu(image, mask, glcm);
    printf("\n");

    glcm_cpu_parallel(image,mask,para_glcm);
    if (check_output(glcm, para_glcm,ng,na))
    {
        printf("Cpu Parallel Result Correct!\n\n");
    }
    else
    {
        printf("Cpu Parallel Result Wrong!\n\n");
    }
    reduce_glcm_gpu(image, mask, reduce_glcm);
    if (check_output(glcm, reduce_glcm,ng,na))
    {
        printf("Reduce Kernel Result Correct!\n\n");
    }
    else
    {
        printf("Reduce Kernel Result Wrong!\n\n");
    }

    atomic_glcm_gpu(image, mask, atomic_glcm);
    if (check_output(glcm, atomic_glcm,ng,na))
    {
        printf("Atomic Kernel Result Correct!\n\n");
    }
    else
    {
        printf("Atomic Kernel Result Wrong!\n\n");
    }
}
*/
