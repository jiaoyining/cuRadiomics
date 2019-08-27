#include"firstorder.h"

int *Calculate_firstorder_rl(const int *image, int *Range, int MASK_VALUE, int *size, int *stride,int Ng, int bin_width, int batch_size)
{
	//START_TIMER(time)
	// getting the max value of the image intensity
	int dev_bin = Ng * bin_width;
	int nbytes_image = sizeof(int) * size[0] * size[1] * batch_size;
    //int nbytes_mask =  sizeof(int) * size[0] * size[1] * batch_size;
    int nbytes_firstorder =  sizeof(int) * dev_bin * batch_size;

    //int *glcm = (int*)malloc(nbytes_glcm);

    int *dev_size  = NULL;
    int *dev_stride = NULL;
    int *dev_P = NULL;
    int *dev_image = NULL;

    cudaMalloc((void**)&dev_P, nbytes_firstorder);
    cudaMalloc((void**)&dev_image, nbytes_image);

	dim3 grids_Pn(1, 1, batch_size);
	dim3 threads_Pn(bin_width, Ng);
	initialize<<<grids_Pn, threads_Pn>>>(dev_P);

    cudaMalloc((void**)&dev_size, sizeof(int) * 2);
    cudaMalloc((void**)&dev_stride, sizeof(int) * 2);

    //ANDLE_ERROR(cudaMemcpy((void*)dev_image, (void*)image, nbytes_image, cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy((void*)dev_mask, (void*)mask, nbytes_mask, cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy((void*)dev_glcm, (void*)glcm, nbytes_glcm, cudaMemcpyHostToDevice));


    cudaMemcpy(dev_size, size, sizeof(int) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_stride, stride, sizeof(int) * 2, cudaMemcpyHostToDevice);

    //printf("copying: ");
    //PRINT_TIME(time)
    //printf("\n");

    dim3 grids(size[0] / 8, size[1] / 8, batch_size);
    dim3 threads(64, 1);

    //START_TIMER(time)
    //START_TIMER(time)

    Preprocessing_image_firstorder<<<grids, threads>>>(dev_image, image, Range[0], Range[1],MASK_VALUE);
    calculate_firstorder_kernel_rl<<<grids, threads>>>(dev_image, dev_P, dev_size, dev_stride, dev_bin);
    cudaDeviceSynchronize();


    cudaFree(dev_stride);
    cudaFree(dev_size);
    cudaFree(dev_image);

    return dev_P;

    //return dev_glcm;
}




void Calculate_firstorder_Property(PROPERTY_fo *Property_fo, float Epsilon,  int bin_width, int Ng, int batch_size)
{
	// getting the range of the image intensity
	int dev_bin = bin_width * Ng;


    if (Property_fo->Np != NULL)
    {
        delete Property_fo->Np;
        Property_fo->Np = NULL;
        //printf("Property_glcm->s != NULL! \n");
    }
    if (Property_fo->Pn != NULL)
    {
        delete Property_fo->Pn;
        Property_fo->Pn = NULL;
        //printf("Property_glcm->Px != NULL! \n");
    }

    if (Property_fo->PF != NULL)
    {
        delete Property_fo->PF;
        Property_fo->PF = NULL;
        //printf("Property_glcm->Px != NULL! \n");
    }

    if (Property_fo->Pf != NULL)
    {
        delete Property_fo->Pf;
        Property_fo->Pf = NULL;
        //printf("Property_glcm->Px != NULL! \n");
    }

    if (Property_fo->pn != NULL)
    {
        delete Property_fo->pn;
        Property_fo->pn = NULL;
        //printf("Property_glcm->Px != NULL! \n");
    }

    if (Property_fo->P10 != NULL)
    {
        delete Property_fo->P10;
        Property_fo->P10 = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }

    if (Property_fo->P25 != NULL)
    {
        delete Property_fo->P25;
        Property_fo->P25 = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }

    if (Property_fo->P50 != NULL)
    {
        delete Property_fo->P50;
        Property_fo->P50 = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }

    if (Property_fo->P75 != NULL)
    {
        delete Property_fo->P75;
        Property_fo->P75 = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }

    if (Property_fo->P90 != NULL)
    {
        delete Property_fo->P90;
        Property_fo->P90 = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }

    if (Property_fo->N1090 != NULL)
    {
        delete Property_fo->N1090;
        Property_fo->N1090 = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }

    if (Property_fo->mP1090 != NULL)
    {
        delete Property_fo->mP1090;
        Property_fo->mP1090 = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }
    if (Property_fo->Pmax != NULL)
    {
        delete Property_fo->Pmax;
        Property_fo->Pmax = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }
    if (Property_fo->Pmin != NULL)
    {
        delete Property_fo->Pmin;
        Property_fo->Pmin = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }
    if (Property_fo->Pm != NULL)
    {
        delete Property_fo->Pm;
        Property_fo->Pm = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }
    if (Property_fo->Pv != NULL)
    {
        delete Property_fo->Pv;
        Property_fo->Pv = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }

    cudaDeviceSynchronize();


	cudaMalloc((void**)&(Property_fo->Np), sizeof(float) * batch_size);
	dim3 grids_Np(1, 1, batch_size);
	dim3 threads_Np(1, 1);
	initialize_tex<<<grids_Np, threads_Np>>>(Property_fo->Np);

	cudaMalloc((void**)&Property_fo->Pn, sizeof(float) * dev_bin * batch_size);
	dim3 grids_Pn(1, 1, batch_size);
	dim3 threads_Pn(Ng, bin_width);
	initialize_tex<<<grids_Pn, threads_Pn>>>(Property_fo->Pn);

	cudaMalloc((void**)&Property_fo->pn, sizeof(float) * Ng * batch_size);
	dim3 grids_pn(1, Ng, batch_size);
	dim3 threads_pn(1, 1);
	initialize_tex<<<grids_pn, threads_pn>>>(Property_fo->pn);

	cudaMalloc((void**)&Property_fo->PF, sizeof(int) * dev_bin * batch_size);
	dim3 grids_PF(1, 1, batch_size);
	dim3 threads_PF(Ng, bin_width);
	initialize<<<grids_PF, threads_PF>>>(Property_fo->PF);


	cudaMalloc((void**)&Property_fo->Pf, sizeof(float) * dev_bin * batch_size);
	dim3 grids_Pf(1, 1, batch_size);
	dim3 threads_Pf(Ng, bin_width);
	initialize_tex<<<grids_Pf, threads_Pf>>>(Property_fo->Pf);

	cudaMalloc((void**)&Property_fo->P25, sizeof(int) * batch_size);
	dim3 grids_P25(1, 1, batch_size);
	dim3 threads_P25(1, 1);
	initialize<<<grids_P25, threads_P25>>>(Property_fo->P25);

	cudaMalloc((void**)&Property_fo->P75, sizeof(int) * batch_size);
	dim3 grids_P75(1, 1, batch_size);
	dim3 threads_P75(1, 1);
	initialize<<<grids_P75, threads_P75>>>(Property_fo->P75);

	cudaMalloc((void**)&Property_fo->P50, sizeof(int) * batch_size);
	dim3 grids_P50(1, 1, batch_size);
	dim3 threads_P50(1, 1);
	initialize<<<grids_P50, threads_P50>>>(Property_fo->P50);

	cudaMalloc((void**)&Property_fo->P90, sizeof(int) * batch_size);
	dim3 grids_P90(1, 1, batch_size);
	dim3 threads_P90(1, 1);
	initialize<<<grids_P90, threads_P90>>>(Property_fo->P90);

	cudaMalloc((void**)&Property_fo->P10, sizeof(int) * batch_size);
	dim3 grids_P10(1, 1, batch_size);
	dim3 threads_P10(1, 1);
	initialize<<<grids_P10, threads_P10>>>(Property_fo->P10);

	cudaMalloc((void**)&Property_fo->N1090, sizeof(int) * batch_size);
	dim3 grids_N1090(1, 1, batch_size);
	dim3 threads_N1090(1, 1);
	initialize<<<grids_N1090, threads_N1090>>>(Property_fo->N1090);

	cudaMalloc((void**)&Property_fo->mP1090, sizeof(float) * batch_size);
	dim3 grids_mP1090(1, 1, batch_size);
	dim3 threads_mP1090(1, 1);
	initialize_tex<<<grids_mP1090, threads_mP1090>>>(Property_fo->mP1090);

	cudaMalloc((void**)&Property_fo->Pmin, sizeof(int) * batch_size);
	dim3 grids_Pmin(1, 1, batch_size);
	dim3 threads_Pmin(1, 1);
	initialize<<<grids_Pmin, threads_Pmin>>>(Property_fo->Pmin);

	cudaMalloc((void**)&Property_fo->Pmax, sizeof(int) * batch_size);
	dim3 grids_Pmax(1, 1, batch_size);
	dim3 threads_Pmax(1, 1);
	initialize<<<grids_Pmax, threads_Pmax>>>(Property_fo->Pmax);

	cudaMalloc((void**)&Property_fo->Pm, sizeof(float) * batch_size);
	dim3 grids_Pm(1, 1, batch_size);
	dim3 threads_Pm(1, 1);
	initialize_tex<<<grids_Pm, threads_Pm>>>(Property_fo->Pm);

	cudaMalloc((void**)&Property_fo->Pv, sizeof(float) * batch_size);
	dim3 grids_Pv(1, 1, batch_size);
	dim3 threads_Pv(1, 1);
	initialize_tex<<<grids_Pv, threads_Pv>>>(Property_fo->Pv);

	cudaDeviceSynchronize();


    dim3 grids(1, 1, batch_size);
    dim3 threads(Ng, bin_width);

    firstorder_Np<<<grids, threads>>>(Property_fo->P, Property_fo->Np, dev_bin);
    //printf("Get Property_glcm sum! \n");
	//HANDLE_ERROR(cudaMemcpy(Property_glcm->s, s, sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
    firstorder_Pn<<<grids, threads>>>(Property_fo->Pn, Property_fo->P, Property_fo->Np, dev_bin);
    //printf("Get Property_glcm Pn! \n");
    cudaDeviceSynchronize();
    firstorder_pn<<<grids, threads>>>(Property_fo->pn, Property_fo->Pn, Ng, dev_bin, bin_width);
    firstorder_PF<<<grids, threads>>>(Property_fo->PF, Property_fo->P, dev_bin);
    cudaDeviceSynchronize();
    firstorder_Pf<<<grids, threads>>>(Property_fo->Pf, Property_fo->PF,  Property_fo->Np, dev_bin);
    cudaDeviceSynchronize();
    firstorder_P10<<<grids, threads>>>(Property_fo->P10, Property_fo->Pf, dev_bin);
    firstorder_P25<<<grids, threads>>>(Property_fo->P25, Property_fo->Pf, dev_bin);
    firstorder_P50<<<grids, threads>>>(Property_fo->P50, Property_fo->Pf, dev_bin);
    firstorder_P75<<<grids, threads>>>(Property_fo->P75, Property_fo->Pf, dev_bin);
    firstorder_P90<<<grids, threads>>>(Property_fo->P90, Property_fo->Pf, dev_bin);
    firstorder_Pmin<<<grids, threads>>>(Property_fo->Pmin, Property_fo->PF, dev_bin);
    firstorder_Pmax<<<grids, threads>>>(Property_fo->Pmax, Property_fo->PF, dev_bin);
    cudaDeviceSynchronize();
    firstorder_N1090<<<grids, threads>>>(Property_fo->N1090, Property_fo->P, Property_fo->P10, Property_fo->P90,  dev_bin);
    cudaDeviceSynchronize();
    firstorder_mP1090<<<grids, threads>>>(Property_fo->mP1090, Property_fo->N1090, Property_fo->Pn, Property_fo->Np, Property_fo->P10, Property_fo->P90,  dev_bin);
    firstorder_Pm<<<grids, threads>>>(Property_fo->Pm, Property_fo->Pn, dev_bin);
    cudaDeviceSynchronize();
    firstorder_Pv<<<grids, threads>>>(Property_fo->Pv, Property_fo->Pm, Property_fo->Pn, dev_bin);

}


void Calculate_firstorder_Texture_rl(PROPERTY_fo *Property_fo, float *texture_fo, float Epsilon, int Ng, int bin_width, int batch_size)
{
	//START_TIMER(time)

	//float *Texture_fo = (float*)malloc(sizeof(float) * 19 * batch_size);
	//printf("Texture_glcm initialized! \n");


	// getting the range of the image intensity
	int dev_bin = bin_width * Ng;


	dim3 grids1(1, 1, batch_size);
	dim3 threads1(18, 1);
	//cudaMalloc((void**)&texture_glcm, sizeof(float) * NA * batch_size);
	initialize_tex<<<grids1, threads1>>>(texture_fo);


	cudaDeviceSynchronize();


	dim3 grids(1, 1, batch_size);
	dim3 threads(Ng, bin_width);


	//printf("getting Texture_glcm CUDA \n");
	//Calculate_GLCM_Texture_glcm_kernel<<<grids, threads>>>(Texture_glcm, Property_glcm, Ng);
	f1_Energy<<<grids, threads>>>(&texture_fo[0 * batch_size], Property_fo->Pn, Property_fo->Np, dev_bin);
	//cudaMemcpy(&Texture_fo[0 * batch_size], &texture_fo[0 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	f3_Entropy<<<grids, threads>>>(&texture_fo[1 * batch_size], Property_fo->pn, Property_fo->Np, dev_bin, bin_width, Epsilon);
	//cudaMemcpy(&Texture_fo[1 * batch_size], &texture_fo[1 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);

	f4_Minimum<<<grids, threads>>>(&texture_fo[2 * batch_size], Property_fo->Pmin, dev_bin);
	//cudaMemcpy(&Texture_fo[2 * batch_size], &texture_fo[2 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);

	f5_TenthPercentile<<<grids, threads>>>(&texture_fo[3 * batch_size], Property_fo->P10, dev_bin);
	//cudaMemcpy(&Texture_fo[3 * batch_size], &texture_fo[3 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);


	f6_NinetiePercentile<<<grids, threads>>>(&texture_fo[4 * batch_size], Property_fo->P90, dev_bin);
	//cudaMemcpy(&Texture_fo[4 * batch_size], &texture_fo[4 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);


	f7_Maximum<<<grids, threads>>>(&texture_fo[5 * batch_size], Property_fo->Pmax, dev_bin);
	//cudaMemcpy(&Texture_fo[5 * batch_size], &texture_fo[5 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);


	f8_Mean<<<grids, threads>>>(&texture_fo[6 * batch_size], Property_fo->Pm, dev_bin);
	//cudaMemcpy(&Texture_fo[6 * batch_size],  &texture_fo[6 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);

	f9_Median<<<grids, threads>>>(&texture_fo[7 * batch_size], Property_fo->P50, dev_bin);
	//cudaMemcpy(&Texture_fo[7 * batch_size], &texture_fo[7 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);

	f10_InterquartileRange<<<grids, threads>>>(&texture_fo[8 * batch_size], Property_fo->P25, Property_fo->P75, dev_bin);
	//cudaMemcpy(&Texture_fo[8 * batch_size], &texture_fo[8 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);

	f11_Range<<<grids, threads>>>(&texture_fo[9 * batch_size], Property_fo->Pmin, Property_fo->Pmax, dev_bin);
	//cudaMemcpy(&Texture_fo[9 * batch_size], &texture_fo[9 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);


	f12_MAD<<<grids, threads>>>(&texture_fo[10 * batch_size], Property_fo->Pn, Property_fo->Pm, Property_fo->Np, dev_bin);
    //cudaMemcpy(&Texture_fo[10 * batch_size], &texture_fo[10 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);


	f13_rMAD<<<grids, threads>>>(&texture_fo[11 * batch_size], Property_fo->N1090, Property_fo->mP1090, Property_fo->P10, Property_fo->P90, Property_fo->Pn, Property_fo->Np, dev_bin);
	//cudaMemcpy(&Texture_fo[11 * batch_size], &texture_fo[11 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	f14_RMS<<<grids, threads>>>(&texture_fo[12 * batch_size], &texture_fo[0 * batch_size], Property_fo->Np, dev_bin);
	//cudaMemcpy(&Texture_fo[12 * batch_size], &texture_fo[12 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);


	f15_StandardDeviation<<<grids, threads>>>(&texture_fo[13 * batch_size], Property_fo->Pv, dev_bin);
	//cudaMemcpy(&Texture_fo[13 * batch_size], &texture_fo[13 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);


	f16_Skewness<<<grids, threads>>>(&texture_fo[14 * batch_size], Property_fo->Pm, Property_fo->Pn, Property_fo->Pv, dev_bin, Epsilon);
	//cudaMemcpy(&Texture_fo[14 * batch_size], &texture_fo[14 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);


	f17_Kurtosis<<<grids, threads>>>(&texture_fo[15 * batch_size], Property_fo->Pm, Property_fo->Pn, Property_fo->Pv, dev_bin, Epsilon);
	//cudaMemcpy(&Texture_fo[15 * batch_size], &texture_fo[15 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);


	f18_Variance<<<grids, threads>>>(&texture_fo[16 * batch_size], Property_fo->Pv, dev_bin);
	//cudaMemcpy(&Texture_fo[16 * batch_size], &texture_fo[16 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);


	f19_Uniformity<<<grids, threads>>>(&texture_fo[17 * batch_size], Property_fo->pn, dev_bin, bin_width);
	//cudaMemcpy(&Texture_fo[17 * batch_size], &texture_fo[17 * batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);

	//f20_Volume<<<grids, threads>>>(&texture_fo[18 * batch_size], Property_fo->Np, dev_bin);
	//cudaMemcpy(&Texture_fo[18 * batch_size], &texture_fo[18* batch_size], sizeof(float) * batch_size, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();


	//STOP_TIMER(time)

	/*
    printf("getting Texture_glcm: \n");
    printf("f1_Energy: %f \n", Texture_fo[0 * batch_size]);
    printf("f3_Entropy: %f \n", Texture_fo[1 * batch_size]);
    printf("f4_Minimum: %f \n", Texture_fo[2 * batch_size]);
    printf("f5_TenthPercentile: %f \n", Texture_fo[3 * batch_size]);
    printf("f6_NinetiethPercentile: %f \n", Texture_fo[4 * batch_size]);
    printf("f7_Maximum: %f \n", Texture_fo[5 * batch_size]);
    printf("f8_Mean: %f \n", Texture_fo[6 * batch_size]);
    printf("f9_Median: %f \n", Texture_fo[7 * batch_size]);
    printf("f10_InterquartileRange: %f \n", Texture_fo[8 * batch_size]);
    printf("f11_Range: %f \n", Texture_fo[9 * batch_size]);
    printf("f12_MAD: %f \n", Texture_fo[10 * batch_size]);
    printf("f13_rMAD: %f \n", Texture_fo[11 * batch_size]);
    printf("f14_RMS: %f \n", Texture_fo[12 * batch_size]);
    printf("f15_StandardDeviation: %f \n", Texture_fo[13 * batch_size]);
    printf("f16_SKewness: %f \n", Texture_fo[14 * batch_size]);
    printf("f17_Kurtosis: %f \n", Texture_fo[15 * batch_size]);
    printf("f18_Variance: %f \n", Texture_fo[16 * batch_size]);
    printf("f19_Uniformity: %f \n", Texture_fo[17 * batch_size]);
    printf("f20_Uniformity: %f \n", Texture_fo[18 * batch_size]);

	printf("CUDA Texture_glcm finished \n");
	*/


	//delete Property_glcm;

	//free(Texture_fo);

}




__global__ void calculate_firstorder_kernel_rl(int *image, int *P, int *dev_size, int *dev_stride, int dev_bin)
{
    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip = blocks * blockDim.x * blockDim.y + threads;

    int j, P_idx, ix, iy;

    int img_ith, ipix;
    img_ith = ip / (dev_size[0] * dev_size[1]);
    ipix = ip % (dev_size[0] * dev_size[1]);

	ix = ipix / dev_stride[0];
	iy = ipix % dev_stride[0];

	P_idx = int(image[ip] > -1) * (image[ip] + img_ith * dev_bin);
	atomicAdd(&P[P_idx], int(1) * int(image[ip] > -1));

}

__global__ void firstorder_Np(int *P, float *Np, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int img_ith, ipix;
    img_ith = ip / dev_bin;
    ipix = ip % dev_bin;

	atomicAdd(&Np[img_ith], float(P[ip]));
}

// histogram of 0-255
__global__ void firstorder_Pn(float *Pn, int *P, float *Np, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int img_ith;
	img_ith = ip / dev_bin;

	atomicExch(&Pn[ip], float(P[ip]) / Np[img_ith]);
}

__global__ void firstorder_pn(float *pn, float *Pn, int dev_ng, int dev_bin, int bin_width){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicAdd(&pn[ipix / bin_width + img_ith * dev_ng], float(Pn[ip]));
}


__global__ void firstorder_PF(int *PF, int *P, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	for(int i = 0; i < dev_bin; i++){
	atomicAdd(&PF[ip], P[i + img_ith * dev_bin] * float(i <= ipix));
	}
}

__global__ void firstorder_Pf(float *Pf, int *PF, float *Np, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicExch(&Pf[ip], float(PF[ip]) / Np[img_ith]);

}

__global__ void firstorder_P25(int *P25,  float *Pf, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicMax(&P25[img_ith], float(ipix + 1) * float(Pf[ip] < 0.25));
}

__global__ void firstorder_P50(int *P50, float *Pf, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicMax(&P50[img_ith], float(ipix + 1) * float(Pf[ip] < 0.5));
}

__global__ void firstorder_P75(int *P75, float *Pf, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicMax(&P75[img_ith], float(ipix + 1) * float(Pf[ip] < 0.75));
}

__global__ void firstorder_P10(int *P10, float *Pf, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicMax(&P10[img_ith], int(ipix + 1) * int(Pf[ip] < 0.1));
}

__global__ void firstorder_P90(int *P90, float *Pf, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicMax(&P90[img_ith], float(ipix + 1) * float(Pf[ip] < 0.9));
}

__global__ void firstorder_Pmin(int *Pmin, int *PF, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicMin(&Pmin[img_ith], PF[ip] > 0? ipix:255);
}


__global__ void firstorder_Pmax(int *Pmax, int *PF, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicMax(&Pmax[img_ith], ipix * int(PF[ip] > 0));
}


__global__ void firstorder_Pm(float *Pm, float *Pn, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicAdd(&Pm[img_ith],  Pn[ip] * float(ipix));
}





__global__ void firstorder_N1090(int *N1090, int *P,  int *P10, int *P90, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicAdd(&N1090[img_ith], P[ip] * int(ipix >= P10[img_ith]) * int(ipix <= P90[img_ith]));
}

__global__ void firstorder_mP1090(float *mP1090, int *N1090, float *Pn, float *Np, int *P10, int *P90, int dev_bin){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int  ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicAdd(&mP1090[img_ith], Pn[ip] * float(ipix) * Np[img_ith] * float(ipix >= P10[img_ith]) * float(ipix <= P90[img_ith]) / float(N1090[img_ith]));
}


__global__ void firstorder_Pv(float *Pv, float *Pm, float *Pn, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicAdd(&Pv[img_ith],  powf(ipix - Pm[img_ith], 2) * Pn[ip]);
}


__global__ void f1_Energy(float *rst, float *Pn, float *Np, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicAdd(&rst[img_ith], powf(float(ipix), 2) * Np[img_ith] * Pn[ip]);

}


__global__ void f3_Entropy(float *rst, float *pn, float *Np, int dev_bin, int bin_width, float Epsilon){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith, ipig;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;
	ipig = ipix / bin_width;

	atomicAdd(&rst[img_ith], -log2f(pn[ipig + img_ith *  dev_bin / bin_width] + Epsilon) * pn[ipig + img_ith *  dev_bin / bin_width] * float(ipix % bin_width == 0));

}

__global__ void f4_Minimum(float *rst, int *Pmin, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicExch(&rst[img_ith], float(Pmin[img_ith]));
}

__global__ void f5_TenthPercentile(float *rst, int *P10, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int img_ith;
	img_ith = ip / dev_bin;

	atomicExch(&rst[img_ith], float(P10[img_ith]));
}

__global__ void f6_NinetiePercentile(float *rst, int *P90, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int img_ith;
	img_ith = ip / dev_bin;

	atomicExch(&rst[img_ith], float(P90[img_ith]));
}

__global__ void f7_Maximum(float *rst, int *Pmax, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicExch(&rst[img_ith], float(Pmax[img_ith]));
}

__global__ void f8_Mean(float *rst, float *Pm, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicExch(&rst[img_ith], Pm[img_ith]);
}


__global__ void f9_Median(float *rst, int *P50, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int img_ith;
	img_ith = ip / dev_bin;

	atomicExch(&rst[img_ith],  float(P50[img_ith]));
}

__global__ void f10_InterquartileRange(float *rst, int *P25, int *P75, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int img_ith;
	img_ith = ip / dev_bin;

	atomicExch(&rst[img_ith],  float(P75[img_ith] - P25[img_ith]));

}

__global__ void f11_Range(float *rst, int *Pmin, int *Pmax, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int img_ith;
	img_ith = ip / dev_bin;

	atomicExch(&rst[img_ith],  float(Pmax[img_ith] - Pmin[img_ith]));

}

__global__ void f12_MAD(float *rst, float *Pn, float *Pm, float *Np, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicAdd(&rst[img_ith],  abs(float(ipix) - Pm[img_ith]) * Pn[ip]);

}

__global__ void f13_rMAD(float *rst, int *N1090, float *mP1090, int *P10, int *P90, float *Pn, float * Np, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicAdd(&rst[img_ith],  float(ipix>=P10[img_ith]) * float(ipix<=P90[img_ith]) * abs(ipix - mP1090[img_ith]) * Pn[ip] * Np[img_ith] / (N1090[img_ith] + 1));

}

__global__ void f14_RMS(float *rst, float *Energy, float *Np, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int img_ith;
	img_ith = ip / dev_bin;

	atomicExch(&rst[img_ith],  sqrt(abs(Energy[img_ith]) / Np[img_ith]));

}

__global__ void f18_Variance(float *rst, float *Pv, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicExch(&rst[img_ith],  Pv[img_ith]);
 //powf(ipix - Pm[img_ith], 2) * Pn[ip]);
}

__global__ void f15_StandardDeviation(float *rst, float *Pv, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int img_ith;
	img_ith = ip / dev_bin;

	atomicExch(&rst[img_ith],  sqrtf(Pv[img_ith]));
}


__global__ void f16_Skewness(float *rst, float *Pm, float *Pn, float *Pv, int dev_bin, float Epsilon){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicAdd(&rst[img_ith],  powf(ipix - Pm[img_ith], 3) * Pn[ip] / (powf(Pv[img_ith], 1.5) + Epsilon));
}


__global__ void f17_Kurtosis(float *rst, float *Pm, float *Pn, float *Pv, int dev_bin, float Epsilon){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;

	atomicAdd(&rst[img_ith],  powf(ipix - Pm[img_ith], 4) * Pn[ip] / (powf(Pv[img_ith], 2) + Epsilon));

}

__global__ void f19_Uniformity(float *rst, float *pn, int dev_bin, int bin_width){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipig, ipix, img_ith;
	img_ith = ip / dev_bin;
	ipix = ip % dev_bin;
	ipig = ipix / bin_width;

	atomicAdd(&rst[img_ith],  powf(pn[ipig + img_ith * dev_bin / bin_width], 2) * float(ipix % bin_width == 0));

}

__global__ void f20_Volume(float *rst, float *Np, int dev_bin){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ipig, ipix, img_ith;
	img_ith = ip / dev_bin;

	atomicExch(&rst[img_ith],  Np[img_ith]);

}







