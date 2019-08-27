
#include"glcm.h"


/* Calculating GLCM */
int *Calculate_GLCM_rl(const int *image, int *size, int *stride, int *angles, int *Range, int MASK_VALUE, int bin_width, int Ng, int NA, int batch_size)
{
	//START_TIMER(time)
	int nbytes_image = sizeof(int) * size[0] * size[1] * batch_size;
    //int nbytes_mask =  sizeof(int) * size[0] * size[1] * batch_size;
    int nbytes_glcm =  sizeof(int) * Ng * Ng * NA * batch_size;

    //int *glcm = (int*)malloc(nbytes_glcm);

    int *dev_image = NULL;
    int *dev_angles = NULL;
    int *dev_size  = NULL;
    int *dev_stride = NULL;
    int *dev_glcm = NULL;

    cudaMalloc((void**)&dev_image, nbytes_image);
    //HANDLE_ERROR(cudaMalloc((void**)&dev_mask, nbytes_mask));
    cudaMalloc((void**)&dev_glcm, nbytes_glcm);

	dim3 grids_Pn(Ng/8, Ng/8, batch_size);
	dim3 threads_Pn(64, NA);
	initialize<<<grids_Pn, threads_Pn>>>(dev_glcm);

    cudaMalloc((void**)&dev_size, sizeof(int) * 2);
    cudaMalloc((void **)&dev_angles, sizeof(int) * 8);
    cudaMalloc((void**)&dev_stride, sizeof(int) * 2);

    //ANDLE_ERROR(cudaMemcpy((void*)dev_image, (void*)image, nbytes_image, cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy((void*)dev_mask, (void*)mask, nbytes_mask, cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy((void*)dev_glcm, (void*)glcm, nbytes_glcm, cudaMemcpyHostToDevice));


    cudaMemcpy(dev_size, size, sizeof(int) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_stride, stride, sizeof(int) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_angles, angles, sizeof(int) * 8, cudaMemcpyHostToDevice);

    //printf("copying: ");
    //PRINT_TIME(time)
    //printf("\n");

    dim3 grids(16, 16, batch_size);
    dim3 threads(size[0]/16, size[1]/16);


    //START_TIMER(time)
    //START_TIMER(time)
    Preprocessing_image_GLCM<<<grids, threads>>>(dev_image, image, Range[0], Range[1], bin_width, MASK_VALUE);
    //initialize<<<grids1, threads1>>>(dev_glcm);
    cudaDeviceSynchronize();
    calculate_glcm_kernel_rl<<<grids, threads>>>(dev_image, dev_glcm, dev_size, dev_stride, dev_angles, Ng, NA);
    //STOP_TIMER(time)
    //printf("GLCM \n");
    //PRINT_TIME(time)

    //cudaDeviceSynchronize();
    //HANDLE_ERROR(cudaMemcpy(glcm, dev_glcm, nbytes_glcm, cudaMemcpyDeviceToHost));
    //printf("GLCM calculated \n");

    cudaFree(dev_image);
    //HANDLE_ERROR(cudaFree(dev_mask));
    //HANDLE_ERROR(cudaFree(dev_glcm));
    cudaFree(dev_angles);
    cudaFree(dev_stride);
    cudaFree(dev_size);

    return dev_glcm;

    //return dev_glcm;
}




void Calculate_GLCM_Property(PROPERTY_glcm *Property_glcm, float Epsilon, int Ng, int NA, int batch_size)
{
	/*
	PROPERTY_glcm *Property_glcm = (PROPERTY_glcm*)malloc(sizeof(PROPERTY_glcm));
	Property_glcm->P = glcm;
	Property_glcm->s = (float*)malloc(sizeof(float) * NA);
	Property_glcm->Pn = (float*)malloc(sizeof(float) * Ng * Ng * NA);
	Property_glcm->Px = (float*)malloc(sizeof(float) * Ng * NA);
	Property_glcm->Py = (float*)malloc(sizeof(float) * Ng * NA);
	Property_glcm->ux = (float*)malloc(sizeof(float) * NA);
	Property_glcm->uy = (float*)malloc(sizeof(float) * NA);
	Property_glcm->Dx = (float*)malloc(sizeof(float) * NA);
	Property_glcm->Dy = (float*)malloc(sizeof(float) * NA);
	Property_glcm->Pxay = (float*)malloc(sizeof(float) * Ng * NA * 2);
	Property_glcm->Pxsy = (float*)malloc(sizeof(float) * Ng * NA);
	Property_glcm->HX = (float*)malloc(sizeof(float) * NA);
	Property_glcm->HY = (float*)malloc(sizeof(float) * NA);
	Property_glcm->HXY = (float*)malloc(sizeof(float) * NA);
	Property_glcm->HXY1 = (float*)malloc(sizeof(float) * NA);
	Property_glcm->HXY2 = (float*)malloc(sizeof(float) * NA);
	*/

	//START_TIMER(time)
	int nbytes_glcm =  sizeof(int) * Ng * Ng * NA * batch_size;
	//int *P_matrix;
	//cudaMalloc((void**)&P_matrix, nbytes_glcm);
	//cudaMemcpy((void*)P_matrix, (void*)glcm, nbytes_glcm, cudaMemcpyHostToDevice);

	//PROPERTY_glcm Property_glcm;
	//cudaMalloc((void**)&Property_glcm, sizeof(PROPERTY_glcm));

	//cudaMalloc((void**)&(Property_glcm.P), sizeof(int) * NA * Ng * Ng * batch_size);
	//Property_glcm->P = glcm;
    if (Property_glcm->Pn != NULL)
    {
        delete Property_glcm->Pn;
        Property_glcm->Pn = NULL;
        //printf("Property_glcm->Pn != NULL! \n");
    }
    if (Property_glcm->s != NULL)
    {
        delete Property_glcm->s;
        Property_glcm->s = NULL;
        //printf("Property_glcm->s != NULL! \n");
    }
    if (Property_glcm->Px != NULL)
    {
        delete Property_glcm->Px;
        Property_glcm->Px = NULL;
        //printf("Property_glcm->Px != NULL! \n");
    }
    if (Property_glcm->Py != NULL)
    {
        delete Property_glcm->Py;
        Property_glcm->Py = NULL;
        //printf("Property_glcm->Py != NULL! \n");
    }
    if (Property_glcm->ux != NULL)
    {
    	Property_glcm->ux;
    	Property_glcm->ux = NULL;
    	//printf("Property_glcm->ux != NULL! \n");
    }
    if (Property_glcm->uy != NULL)
    {
        delete Property_glcm->uy;
        Property_glcm->uy = NULL;
        //printf("Property_glcm->uy != NULL! \n");
    }
    if (Property_glcm->Dx != NULL)
    {
        delete Property_glcm->Dx;
        Property_glcm->Dx = NULL;
        //printf("Property_glcm->Dx != NULL! \n");
    }
    if (Property_glcm->Dy != NULL)
    {
        delete Property_glcm->Dy;
        Property_glcm->Dy = NULL;
        //printf("Property_glcm->Dy != NULL! \n");
    }
    if (Property_glcm->Pxay != NULL)
    {
        delete Property_glcm->Pxay;
        Property_glcm->Pxay = NULL;
        //printf("Property_glcm->Pxay != NULL! \n");
    }
    if (Property_glcm->Pxsy != NULL)
    {
        delete Property_glcm->Pxsy;
        Property_glcm->Pxsy = NULL;
        //printf("Property_glcm->Pxsy != NULL! \n");
    }
    if (Property_glcm->HX != NULL)
    {
        delete Property_glcm->HX;
        Property_glcm->HX = NULL;
        //printf("Property_glcm->HX != NULL! \n");
    }
    if (Property_glcm->HY != NULL)
    {
        delete Property_glcm->HY;
        Property_glcm->HY = NULL;
        //printf("Property_glcm->HY != NULL! \n");
    }
    if (Property_glcm->HXY != NULL)
    {
        delete Property_glcm->HXY;
        Property_glcm->HXY = NULL;
        //printf("Property_glcm->HXY != NULL! \n");
    }
    if (Property_glcm->HXY1 != NULL)
    {
        delete Property_glcm->HXY1;
        Property_glcm->HXY1 = NULL;
        //printf("Property_glcm->HXY1 != NULL! \n");
    }
    if (Property_glcm->HXY2 != NULL)
    {
        delete Property_glcm->HXY2;
        Property_glcm->HXY2 = NULL;
        //printf("Property_glcm->HXY2 != NULL! \n");
    }
	if (Property_glcm->maxp != NULL)
	{
		delete Property_glcm->maxp;
		Property_glcm->maxp = NULL;
		//printf("Property_glcm->maxp != NULL! \n");
	}

	cudaDeviceSynchronize();


	cudaMalloc((void**)&(Property_glcm->s), sizeof(float) * NA * batch_size);
	dim3 grids_s(1, 1, batch_size);
	dim3 threads_s(1, NA);
	initialize_tex<<<grids_s, threads_s>>>(Property_glcm->s);

	cudaMalloc((void**)&Property_glcm->Pn, sizeof(float) * Ng * Ng * NA * batch_size);
	dim3 grids_Pn(Ng/8, Ng/8, batch_size/4);
	dim3 threads_Pn(16, 16);
	initialize_tex<<<grids_Pn, threads_Pn>>>(Property_glcm->Pn);

	cudaMalloc((void**)&Property_glcm->Px, sizeof(float) * Ng * NA * batch_size);
	dim3 grids_Px(Ng, 1, batch_size);
	dim3 threads_Px(1, NA);
	initialize_tex<<<grids_Px, threads_Px>>>(Property_glcm->Px);

	cudaMalloc((void**)&Property_glcm->Py, sizeof(float) * Ng * NA * batch_size);
	dim3 grids_Py(Ng, 1, batch_size);
	dim3 threads_Py(1, NA);
	initialize_tex<<<grids_Py, threads_Py>>>(Property_glcm->Py);

	cudaMalloc((void**)&Property_glcm->Pxay, sizeof(float) * Ng * NA * 2 * batch_size);
	dim3 grids_Pxay(Ng, 2, batch_size);
	dim3 threads_Pxay(1, NA);
	initialize_tex<<<grids_Pxay, threads_Pxay>>>(Property_glcm->Pxay);


	cudaMalloc((void**)&Property_glcm->Pxsy, sizeof(float) * Ng * NA * batch_size);
	dim3 grids_Pxsy(Ng, 1, batch_size);
	dim3 threads_Pxsy(1, NA);
	initialize_tex<<<grids_Pxsy, threads_Pxsy>>>(Property_glcm->Pxsy);

	cudaMalloc((void**)&Property_glcm->ux, sizeof(float) * NA * batch_size);
	dim3 grids_ux(1, 1, batch_size);
	dim3 threads_ux(1, NA);
	initialize_tex<<<grids_ux, threads_ux>>>(Property_glcm->ux);

	cudaMalloc((void**)&Property_glcm->uy, sizeof(float) * NA * batch_size);
	dim3 grids_uy(1, 1, batch_size);
	dim3 threads_uy(1, NA);
	initialize_tex<<<grids_uy, threads_uy>>>(Property_glcm->uy);

	cudaMalloc((void**)&Property_glcm->Dx, sizeof(float) * NA * batch_size);
	dim3 grids_Dx(1, 1, batch_size);
	dim3 threads_Dx(1, NA);
	initialize_tex<<<grids_Dx, threads_Dx>>>(Property_glcm->Dx);

	cudaMalloc((void**)&Property_glcm->Dy, sizeof(float) * NA * batch_size);
	dim3 grids_Dy(1, 1, batch_size);
	dim3 threads_Dy(1, NA);
	initialize_tex<<<grids_Dy, threads_Dy>>>(Property_glcm->Dy);

	cudaMalloc((void**)&Property_glcm->HX, sizeof(float) * NA * batch_size);
	dim3 grids_HX(1, 1, batch_size);
	dim3 threads_HX(1, NA);
	initialize_tex<<<grids_HX, threads_HX>>>(Property_glcm->HX);

	cudaMalloc((void**)&Property_glcm->HY, sizeof(float) * NA * batch_size);
	dim3 grids_HY(1, 1, batch_size);
	dim3 threads_HY(2, 2);
	initialize_tex<<<grids_HY, threads_HY>>>(Property_glcm->HY);

	cudaMalloc((void**)&Property_glcm->HXY, sizeof(float) * NA * batch_size);
	dim3 grids_HXY(1, 1, batch_size);
	dim3 threads_HXY(2, 2);
	initialize_tex<<<grids_HXY, threads_HXY>>>(Property_glcm->HXY);

	cudaMalloc((void**)&Property_glcm->HXY1, sizeof(float) *NA * batch_size);
	dim3 grids_HXY1(1, 1, batch_size);
	dim3 threads_HXY1(1, NA);
	initialize_tex<<<grids_s, threads_s>>>(Property_glcm->HXY1);

	cudaMalloc((void**)&Property_glcm->HXY2, sizeof(float) * NA * batch_size);
	dim3 grids_HXY2(1, 1, batch_size);
	dim3 threads_HXY2(1, NA);
	initialize_tex<<<grids_HXY2, threads_HXY2>>>(Property_glcm->HXY2);

	cudaMalloc((void**)&Property_glcm->maxp, sizeof(int) * NA * batch_size);
	dim3 grids_maxp(1, 1, batch_size);
	dim3 threads_maxp(1, NA);
	initialize<<<grids_maxp, threads_maxp>>>(Property_glcm->maxp);

	cudaMalloc((void**)&Property_glcm->DA, sizeof(float) * NA * batch_size);
	dim3 grids_DA(1, 1, batch_size);
	dim3 threads_DA(1, NA);
	initialize_tex<<<grids_DA, threads_DA>>>(Property_glcm->DA);

	cudaDeviceSynchronize();
	//printf("Property_glcm initialized! \n");

    dim3 grids(Ng/8, Ng/8, batch_size);
    dim3 threads(64, NA);


    //printf("getting property_glcm CUDA \n");


    //Calculate_GLCM_Property_glcm_kernel<<<grids, threads>>>(Property_glcm, P_matrix, NA, Ng, Epsilon);
    //Property_glcm->epsilon = Epsilon;
    //Property_glcm.P = glcm;
    //printf("P %d \n", Property_glcm.P[10000]);
    //printf("Get Property_glcm P! \n");

    GLCM_sum<<<grids, threads>>>(Property_glcm->P, Property_glcm->s, Ng, NA);
    GLCM_Pn<<<grids, threads>>>(Property_glcm->P, Property_glcm->Pn, Property_glcm->s, Ng, NA, Epsilon);
    GLCM_Property1<<<grids, threads>>>(
    		Property_glcm->Pn,
    		Property_glcm->Px,
    		Property_glcm->Py,
    		Property_glcm->ux,
    		Property_glcm->uy,
    		Property_glcm->Pxay,
    		Property_glcm->Pxsy,
    		Ng,
			NA,
			Epsilon);
    GLCM_Property2<<<grids, threads>>>(Property_glcm->P,
    		Property_glcm->s,
    		Property_glcm->Pn,
    		Property_glcm->Px,
    		Property_glcm->Py,
    		Property_glcm->ux,
    		Property_glcm->uy,
    		Property_glcm->Dx,
    		Property_glcm->Dy,
    		Property_glcm->Pxay,
    		Property_glcm->Pxsy,
    		Property_glcm->HX,
    		Property_glcm->HY,
    		Property_glcm->HXY,
    		Property_glcm->HXY1,
    		Property_glcm->HXY2,
    		Property_glcm->maxp,
    		Property_glcm->DA,
    		Ng,
			NA,
			Epsilon);
    	/*
    GLCM_Px<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->Px, Ng, NA);
    //printf("Get Property_glcm Px! \n");
    cudaDeviceSynchronize();
    GLCM_Py<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->Py, Ng, NA);
    //printf("Get Property_glcm Py! \n");
    cudaDeviceSynchronize();
    GLCM_ux<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->ux, Ng, NA);
    //printf("Get Property_glcm ux! \n");
    cudaDeviceSynchronize();
    GLCM_uy<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->uy, Ng, NA);
    //printf("Get Property_glcm uy! \n");
    cudaDeviceSynchronize();
    GLCM_Dx<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->ux, Property_glcm->Dx, Ng, NA);
    //printf("Get Property_glcm Dx! \n");
    cudaDeviceSynchronize();
    GLCM_Dy<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->uy, Property_glcm->Dy, Ng, NA);
    //printf("Get Property_glcm Dy! \n");
    cudaDeviceSynchronize();
    //printf("Get Property_glcm Dx! \n");
    GLCM_Pxay<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->Pxay, Ng, NA);
    //printf("Get Property_glcm Pxay! \n");
    cudaDeviceSynchronize();
    GLCM_Pxsy<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->Pxsy, Ng, NA);
    //printf("Get Property_glcm Pxsy! \n");
    cudaDeviceSynchronize();
    //printf("Get Property_glcm Pxay! \n");
    GLCM_HX<<<grids, threads>>>(Property_glcm->Px, Property_glcm->HX, Epsilon, Ng, NA);
    //printf("Get Property_glcm HX! \n");
    cudaDeviceSynchronize();
    GLCM_HY<<<grids, threads>>>(Property_glcm->Py, Property_glcm->HY, Epsilon, Ng, NA);
    //printf("Get Property_glcm HY! \n");
    cudaDeviceSynchronize();
    GLCM_HXY<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->HXY, Epsilon, Ng, NA);
    //printf("Get Property_glcm HXY \n");
    cudaDeviceSynchronize();
    GLCM_HXY1<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->Px, Property_glcm->Py, Property_glcm->HXY1, Epsilon, Ng, NA);
    //printf("Get Property_glcm HXY1! \n");
    cudaDeviceSynchronize();
    GLCM_HXY2<<<grids, threads>>>(Property_glcm->Pn, Property_glcm->Px, Property_glcm->Py, Property_glcm->HXY2, Epsilon, Ng, NA);
    //printf("Get Property_glcm HXY2! \n");
    cudaDeviceSynchronize();
    //printf("Get Property_glcm H! \n");
    GLCM_maxp<<<grids, threads>>>(Property_glcm->P, Property_glcm->maxp, Ng, NA);
    //printf("Get Property_glcm maxp! \n");
    cudaDeviceSynchronize();
    GLCM_DA<<<grids, threads>>>(Property_glcm->DA, Property_glcm->Pxsy, Ng, NA);
    */

	//printf("Get Property_glcm maxp! \n");


    //printf("ux: %d, %d, %d, %d \n", Property_glcm.P[0], Property_glcm.P[1], Property_glcm.P[2], Property_glcm.P[3]);

    //HANDLE_ERROR(cudaDeviceSynchronize());
    //STOP_TIMER(time)
    //printf("getting Property_glcm: ");
    //PRINT_TIME(time)
    //printf("\n");
    //printf("CUDA property_glcm finished \n");
    //cudaFree(glcm);

    //return Property_glcm;
    //HANDLE_ERROR(cudaDeviceSynchronize());
}


void Calculate_GLCM_Texture_rl(PROPERTY_glcm *Property_glcm, float *texture_glcm, float Epsilon, int Ng, int NA, int batch_size)
{
	//START_TIMER(time)

	float *Texture_glcm = (float*)malloc(sizeof(float) * 23 * batch_size);

	//float *texture_glcm = NULL;
	dim3 grids1(1, 1, batch_size);
	dim3 threads1(23, 1);
	//cudaMalloc((void**)&texture_glcm, sizeof(float) * NA * batch_size);
	initialize_tex<<<grids1, threads1>>>(texture_glcm);
	cudaDeviceSynchronize();


	//printf("Texture_glcm initialized! \n");

	dim3 grids(Ng/8, Ng/8, batch_size);
	dim3 threads(64, NA);


	glcm_features<<<grids, threads>>>(texture_glcm,Property_glcm->s,Property_glcm->Pn,Property_glcm->ux,Property_glcm->uy,
									   Property_glcm->Dx, Property_glcm->Dy,Property_glcm->Pxsy,Property_glcm->Pxay,Property_glcm->HX,
									   Property_glcm->HY,Property_glcm->HXY,Property_glcm->HXY1,Property_glcm->HXY2,Property_glcm->maxp,
									   Property_glcm->DA,batch_size,Ng,NA,Epsilon);


	cudaDeviceSynchronize();


	//STOP_TIMER(time)
	/*

    printf("getting Texture_glcm: \n");
    printf("f1_AutoCorrelation: %f \n", Texture_glcm[0 * batch_size + 5]/4);
    printf("f2_JointAverage: %f \n", Texture_glcm[1 * batch_size + 5]/4);
    printf("f3_CLusterProminence: %f \n", Texture_glcm[2 * batch_size + 5]/4);
    printf("f4_ClusterShade: %f \n", Texture_glcm[3 * batch_size + 5]/4);
    printf("f5_ClusterTendency: %f \n", Texture_glcm[4 * batch_size + 5]/4);
    printf("f6_Contrast: %f \n", Texture_glcm[5 * batch_size + 5]/4);
    printf("f7_Correlation: %f \n", Texture_glcm[6 * batch_size + 5]/4);
    printf("f8_DifferenceAverage: %f \n", Texture_glcm[7 * batch_size + 5]/4);
    printf("f9_DifferenceEntropy: %f \n", Texture_glcm[8 * batch_size + 5]/4);
    printf("f10_DifferenceVariance: %f \n", Texture_glcm[9 * batch_size + 5]/4);
    printf("f11_JointEnergy: %f \n", Texture_glcm[10 * batch_size + 5]/4);
    printf("f12_JointEntropy: %f \n", Texture_glcm[11 * batch_size + 5]/4);
    printf("f13_IMC1: %f \n", Texture_glcm[12 * batch_size + 5]/4);
    printf("f14_IMC2: %f \n", Texture_glcm[13 * batch_size + 5]/4);
    printf("f15_IDM: %f \n", Texture_glcm[14 * batch_size + 5]/4);
    printf("f17_IDMN: %f \n", Texture_glcm[15 * batch_size + 5]/4);
    printf("f18_ID: %f \n", Texture_glcm[16 * batch_size + 5]/4);
    printf("f19_IDN: %f \n", Texture_glcm[17 * batch_size +5]/4);
    printf("f20_InverseVariance: %f \n", Texture_glcm[18 * batch_size + 5]/4);
    printf("f21_MaximumProbability: %f \n", Texture_glcm[19 * batch_size + 5]/4);
    printf("f22_SumAverage: %f \n", Texture_glcm[20 * batch_size + 5]/4);
    printf("f23_SumEntropy: %f \n", Texture_glcm[21 * batch_size + 5]/4);
    printf("f24_SumSquares: %f \n", Texture_glcm[22 * batch_size + 5]/4);
    //PRINT_TIME(time)
    printf("\n");

	printf("CUDA Texture_glcm finished \n");
	*/




	//delete Property_glcm;

	free(Texture_glcm);

}


__global__ void GLCM_Property(int *P,
						 float *s,
						 float *Pn,
						 float *Px,
						 float *Py,
						 float *ux,
						 float *uy,
						 float *Dx,
						 float *Dy,
						 float *Pxay,
						 float *Pxsy,
						 float *HX,
						 float *HY,
						 float *HXY,
						 float *HXY1,
						 float *HXY2,
						 int *maxp,
						 float *DA,
						 int Ng,
						 int NA,
						 float epsilon
						 )
{

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ia, ipix, img_ith, ix, iy;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ia = ipix % NA;
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;

	atomicAdd(&s[img_ith * NA + ia], float(P[ip]));
	//cudaDeviceSynchronize();

	atomicExch(&Pn[ip], float(P[ip])/(s[img_ith * NA + ia] + epsilon));
	//cudaDeviceSynchronize();

	atomicAdd(&Px[img_ith * Ng * NA + ix * NA + ia], Pn[ip]);
	atomicAdd(&Py[img_ith * Ng * NA + iy * NA + ia], Pn[ip]);
	atomicAdd(&Pxay[img_ith * 2 * Ng * NA + ix * NA + iy * NA + ia], Pn[ip]);
	atomicAdd(&Pxsy[img_ith * Ng * NA + abs(ix - iy) * NA + ia], Pn[ip]);
	atomicAdd(&ux[img_ith * NA + ia], Pn[ip] * (ix + 1));
	atomicAdd(&uy[img_ith * NA + ia], Pn[ip] * (iy + 1));
	//cudaDeviceSynchronize();

	atomicAdd(&Dx[img_ith * NA + ia], powf(ix + 1 - ux[img_ith * NA + ia], 2) * Pn[ip]);
	atomicAdd(&Dy[img_ith * NA + ia], powf(iy + 1 - uy[img_ith * NA + ia], 2) * Pn[ip]);
	//cudaDeviceSynchronize();


	atomicAdd(&HX[img_ith * NA + ia], float(iy==0) * (-Px[img_ith * Ng * NA + ix * NA + ia] * log2f(Px[img_ith * Ng * NA + ix * NA + ia] + epsilon)));
	atomicAdd(&HY[img_ith * NA + ia], float(ix==0) * (-Py[img_ith * NA * Ng + iy * NA + ia] * log2f(Py[img_ith * NA * Ng + iy * NA + ia] + epsilon)));
	atomicAdd(&HXY[img_ith * NA + ia], -Pn[ip] * log2f(Pn[ip] + epsilon));
	atomicAdd(&HXY1[img_ith * NA + ia], -Pn[ip] * log2f(Px[img_ith * NA * Ng + ix * NA + ia] * Py[img_ith * NA * Ng + iy * NA + ia] + epsilon));
	atomicAdd(&HXY2[img_ith * NA + ia],
			-Px[img_ith * NA * Ng + ix * NA + ia]
			* Py[img_ith * NA * Ng + iy * NA + ia]
			* log2f(Px[img_ith * NA * Ng + ix * NA + ia] * Py[img_ith * NA * Ng + iy * NA + ia] + epsilon));
	atomicMax(&maxp[img_ith * NA + ia], P[ip]);
	atomicAdd(&DA[img_ith * NA + ia], float(ix) * Pxsy[img_ith * NA * Ng + ix * NA + ia] * float(iy == 0));


}
__global__ void GLCM_Property2(int *P,
						 float *s,
						 float *Pn,
						 float *Px,
						 float *Py,
						 float *ux,
						 float *uy,
						 float *Dx,
						 float *Dy,
						 float *Pxay,
						 float *Pxsy,
						 float *HX,
						 float *HY,
						 float *HXY,
						 float *HXY1,
						 float *HXY2,
						 int *maxp,
						 float *DA,
						 int Ng,
						 int NA,
						 float epsilon
						 )
{

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ia, ipix, img_ith, ix, iy;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ia = ipix % NA;
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;


	atomicAdd(&Dx[img_ith * NA + ia], powf(ix + 1 - ux[img_ith * NA + ia], 2) * Pn[ip]);
	atomicAdd(&Dy[img_ith * NA + ia], powf(iy + 1 - uy[img_ith * NA + ia], 2) * Pn[ip]);
	//cudaDeviceSynchronize();

	atomicAdd(&HX[img_ith * NA + ia], float(iy==0) * (-Px[img_ith * Ng * NA + ix * NA + ia] * log2f(Px[img_ith * Ng * NA + ix * NA + ia] + epsilon)));
	atomicAdd(&HY[img_ith * NA + ia], float(ix==0) * (-Py[img_ith * NA * Ng + iy * NA + ia] * log2f(Py[img_ith * NA * Ng + iy * NA + ia] + epsilon)));
	atomicAdd(&HXY[img_ith * NA + ia], -Pn[ip] * log2f(Pn[ip] + epsilon));
	atomicAdd(&HXY1[img_ith * NA + ia], -Pn[ip] * log2f(Px[img_ith * NA * Ng + ix * NA + ia] * Py[img_ith * NA * Ng + iy * NA + ia] + epsilon));
	atomicAdd(&HXY2[img_ith * NA + ia],
			-Px[img_ith * NA * Ng + ix * NA + ia]
			* Py[img_ith * NA * Ng + iy * NA + ia]
			* log2f(Px[img_ith * NA * Ng + ix * NA + ia] * Py[img_ith * NA * Ng + iy * NA + ia] + epsilon));
	atomicMax(&maxp[img_ith * NA + ia], P[ip]);
	atomicAdd(&DA[img_ith * NA + ia], float(ix) * Pxsy[img_ith * NA * Ng + ix * NA + ia] * float(iy == 0));


}
__global__ void GLCM_Property1(
						 float *Pn,
						 float *Px,
						 float *Py,
						 float *ux,
						 float *uy,
						 float *Pxay,
						 float *Pxsy,
						 int Ng,
						 int NA,
						 float epsilon
						 )
{

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ia, ipix, img_ith, ix, iy;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ia = ipix % NA;
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;


	atomicAdd(&Px[img_ith * Ng * NA + ix * NA + ia], Pn[ip]);
	atomicAdd(&Py[img_ith * Ng * NA + iy * NA + ia], Pn[ip]);
	atomicAdd(&Pxay[img_ith * 2 * Ng * NA + ix * NA + iy * NA + ia], Pn[ip]);
	atomicAdd(&Pxsy[img_ith * Ng * NA + abs(ix - iy) * NA + ia], Pn[ip]);
	atomicAdd(&ux[img_ith * NA + ia], Pn[ip] * (ix + 1));
	atomicAdd(&uy[img_ith * NA + ia], Pn[ip] * (iy + 1));
	//cudaDeviceSynchronize();

}

__global__ void calculate_glcm_kernel_rl(int *image, int *glcm, int *dev_size, int *dev_stride, int *dev_angles, int dev_ng, int dev_na)
{
    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip = blocks * blockDim.x * blockDim.y + threads;

    int j, glcm_idx, a, iz, iy, ix;

    int img_ith, ipix;
    img_ith = ip / (dev_size[0] * dev_size[1]);
    ipix = ip % (dev_size[0] * dev_size[0]);

	ix = ipix / dev_stride[0];
	iy = ipix % dev_stride[0];

	for (a = 0; a < dev_na; a++)
	{
		if (ix + dev_angles[a * 2] >= 0 && ix + dev_angles[a * 2] < dev_size[0] &&
			iy + dev_angles[a * 2 + 1] >= 0 && iy + dev_angles[a * 2 + 1] < dev_size[1])
		{
			j = ip + dev_angles[a * 2] * dev_stride[0] + dev_angles[a * 2 + 1] * dev_stride[1];
			glcm_idx = int(image[ip] > -1) * int(image[j] > -1) * (a + image[j] * dev_na + image[ip] * dev_na * dev_ng + img_ith * dev_ng * dev_ng * dev_na);
			atomicAdd(&glcm[glcm_idx], 1 * int(image[ip] > -1) * int(image[j] > -1));
		}
	}
}

__global__ void GLCM_sum(int *P, float *s, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ia = ipix % NA;

	atomicAdd(&s[img_ith * NA + ia], float(P[ip]));
}

__global__ void GLCM_Pn(int *P, float *Pn, float *sum, int Ng, int NA, float epsilon){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ia = ipix % NA;
	atomicExch(&Pn[ip], float(P[ip])/(sum[img_ith * NA + ia] + epsilon));
}



__global__ void GLCM_Px(float *P, float *Px, int Ng, int NA){


    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

    atomicAdd(&Px[img_ith * Ng * NA + ix * NA + ia], P[ip]);

}

__global__ void GLCM_Py(float *P, float *Py, int Ng, int NA){


    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

    atomicAdd(&Py[img_ith * Ng * NA + iy * NA + ia], P[ip]);

}


__global__ void GLCM_ux(float *P, float *ux, int Ng, int NA){

    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    ia = ipix % NA;

    atomicAdd(&ux[img_ith * NA + ia], P[ip] * (ix + 1));
}

__global__ void GLCM_uy(float *P, float *uy, int Ng, int NA){

    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip= blocks * blockDim.x * blockDim.y + threads;

    int iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    iy = ipix / NA % Ng;
    ia = ipix % NA;

    atomicAdd(&uy[img_ith * NA + ia], P[ip] * (iy + 1));
}

__global__ void GLCM_Dx(float *P,  float *ux, float *Dx, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ix, ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ix = ipix / NA / Ng;
	ia = ipix % NA;

	atomicAdd(&Dx[img_ith * NA + ia], powf(ix + 1 - ux[img_ith * NA + ia], 2) * P[ip]);
	//atomicExch(&Dx[img_ith * NA + ia], sqrtf(Dx[img_ith * NA + ia]));
}

__global__ void GLCM_Dy(float *P,  float *uy, float *Dy, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int iy, ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	iy = ipix / NA % Ng;
	ia = ipix % NA;

	atomicAdd(&Dy[img_ith * NA + ia], powf(iy + 1 - uy[img_ith * NA + ia], 2) * P[ip]);
	//atomicExch(&Dy[img_ith * NA + ia], sqrtf(Dy[img_ith * NA + ia]));
}

__global__ void GLCM_Pxay(float *P, float *Pxay, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ix, iy, ia, img_ith, ipix;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ix = ipix / NA / Ng;
	iy = ipix / NA % Ng;
	ia = ipix % NA;

	atomicAdd(&Pxay[img_ith * 2 * Ng * NA + ix * NA + iy * NA + ia], P[ip]);

}

__global__ void GLCM_Pxsy(float *P, float *Pxsy, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ix, iy, ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ix = ipix / NA / Ng;
	iy = ipix / NA % Ng;
	ia = ipix % NA;

	atomicAdd(&Pxsy[img_ith * Ng * NA + abs(ix - iy) * NA + ia], P[ip]);

}


__global__  void GLCM_HX(float *Px, float *HX, float epsilon, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	//for(i = 0; i< Ng; i++)
	//{HX[0] -= Px[ipix] * log2f(Px[ipix] + epsilon);}
	int ix, iy, ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ix = ipix / NA / Ng;
	iy = ipix / NA % Ng;
	ia = ipix % NA;

	atomicAdd(&HX[img_ith * NA + ia], float(iy==0) * (-Px[img_ith * Ng * NA + ix * NA + ia] * log2f(Px[img_ith * Ng * NA + ix * NA + ia] + epsilon)));
	//atomicExch(&HX[0], sum);
}

__global__  void GLCM_HY(float *Py, float *HY, float epsilon, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	//for(i = 0; i< Ng; i++)
	//{HX[0] -= Px[ipix] * log2f(Px[ipix] + epsilon);}
	int ix, iy, ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ix = ipix / NA / Ng;
	iy = ipix / NA % Ng;
	ia = ipix % NA;

	atomicAdd(&HY[img_ith * NA + ia], float(ix==0) * (-Py[img_ith * NA * Ng + iy * NA + ia] * log2f(Py[img_ith * NA * Ng + iy * NA + ia] + epsilon)));
	//atomicExch(&HX[0], sum);
}

__global__  void GLCM_HXY(float *P, float *HXY, float epsilon, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ia = ipix % NA;

	atomicAdd(&HXY[img_ith * NA + ia], -P[ip] * log2f(P[ip] + epsilon));
	//atomicExch(&HX[0], sum);

}

__global__ void GLCM_HXY1(float *P, float *Px, float *Py, float *HXY1, float epsilon, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ix, iy, ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ix = ipix / NA / Ng;
	iy = ipix / NA % Ng;
	ia = ipix % NA;

	atomicAdd(&HXY1[img_ith * NA + ia], -P[ip] * log2f(Px[img_ith * NA * Ng + ix * NA + ia] * Py[img_ith * NA * Ng + iy * NA + ia] + epsilon));
	//atomicExch(&HXY1[0], sum);

}

__global__ void GLCM_HXY2(float *P, float *Px, float *Py, float *HXY2, float epsilon, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ix, iy, ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ix = ipix / NA / Ng;
	iy = ipix / NA % Ng;
	ia = ipix % NA;

	atomicAdd(&HXY2[img_ith * NA + ia],
			-Px[img_ith * NA * Ng + ix * NA + ia]
			* Py[img_ith * NA * Ng + iy * NA + ia]
			* log2f(Px[img_ith * NA * Ng + ix * NA + ia] * Py[img_ith * NA * Ng + iy * NA + ia] + epsilon));
	//atomicExch(&HXY2[0], sum);

}

__global__ void GLCM_maxp(int *P, int *maxp, int Ng, int NA){
	//float dst[4];
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

    atomicMax(&maxp[img_ith * NA + ia], P[ip]);

}

__global__ void GLCM_DA(float *DA, float *Pxsy, int Ng, int NA) {

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&DA[img_ith * NA + ia], float(ix) * Pxsy[img_ith * NA * Ng + ix * NA + ia] * float(iy == 0));
	//*rst /= NA;
}

/* FEATURE EXTRACTION*/
__global__ void glcm_features(float *rst,
								   float *s,
								   float *Pn,
								   float *ux,
								   float *uy,
								   float *Dx,
								   float *Dy,
								   float *Pxsy,
								   float *Pxay,
								   float *HX,
								   float *HY,
								   float *HXY,
								   float *HXY1,
								   float *HXY2,
								   int *maxp,
								   float *DA,
								   int batch_size,
								   int Ng,
								   int NA,
								   float epsilon){

    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

    atomicAdd(&rst[0 * batch_size + img_ith], Pn[ip] * (ix + 1) * (iy + 1) / NA);
    atomicAdd(&rst[1 * batch_size + img_ith], Pn[ip] * float(ix + 1) / NA);
    atomicAdd(&rst[2 * batch_size + img_ith], powf((float(ix + 1) + float(iy + 1) - ux[img_ith * NA + ia] - uy[img_ith * NA + ia]), 4) * Pn[ip] / NA);
    atomicAdd(&rst[3 * batch_size + img_ith], powf((float(ix + 1) + float(iy + 1) - ux[img_ith * NA + ia] - uy[img_ith * NA + ia]), 3) * Pn[ip] / NA);
    atomicAdd(&rst[4 * batch_size + img_ith], powf((float(ix + 1) + float(iy + 1) - ux[img_ith * NA + ia] - uy[img_ith * NA + ia]), 2) * Pn[ip] / NA);
    atomicAdd(&rst[5 * batch_size + img_ith], powf((ix - iy), 2) * Pn[ip] / NA);
    atomicAdd(&rst[6 * batch_size + img_ith], Pn[ip] * (ix + 1 - ux[img_ith * NA + ia]) * (iy + 1 - uy[img_ith * NA + ia]) /(sqrtf(Dx[img_ith * NA + ia] * Dy[img_ith * NA + ia]) + epsilon) /NA);
    atomicAdd(&rst[7 * batch_size + img_ith], float(ix) * Pxsy[img_ith * NA * Ng  + ix * NA + ia] * float(iy == 0) / NA);
    atomicAdd(&rst[8 * batch_size + img_ith], -Pxsy[img_ith * NA * Ng + ix * NA + ia] * log2f(Pxsy[img_ith * NA * Ng + ix * NA + ia] + epsilon) * float(iy == 0) / NA);
    atomicAdd(&rst[9 * batch_size + img_ith], powf(float(ix) - DA[img_ith * NA + ia], 2) * Pxsy[img_ith * Ng * NA + ix * NA + ia] * float(iy == 0) / NA);
    atomicAdd(&rst[10 * batch_size + img_ith], powf(Pn[ip], 2) / NA);
    atomicAdd(&rst[11 * batch_size + img_ith], -Pn[ip] * log2f(Pn[ip] + epsilon) / NA);
    atomicAdd(&rst[12 * batch_size + img_ith], float(ix == 0) * float(iy == 0) * (HXY[img_ith * NA + ia] - HXY1[img_ith * NA + ia]) / max(HX[img_ith * NA + ia], HY[img_ith * NA + ia]) / NA);
    atomicAdd(&rst[13 * batch_size + img_ith], float(ix == 0) * float(iy == 0) * sqrtf(abs(1 - powf(M_E, -2 * (HXY2[img_ith * NA + ia] - HXY[img_ith * NA + ia])))) / NA);
    atomicAdd(&rst[14 * batch_size + img_ith], Pxsy[img_ith * NA * Ng + ix * NA + ia] / (1 + powf(ix, 2)) * float(iy == 0) / NA);
    atomicAdd(&rst[15 * batch_size + img_ith], Pxsy[img_ith * NA * Ng + ix * NA + ia] / (1 + (powf(ix, 2)/powf(Ng, 2))) * float(iy == 0) / NA);
    atomicAdd(&rst[16 * batch_size + img_ith], Pxsy[img_ith * NA * Ng + ix * NA + ia] / float(1 + ix) * float(iy == 0) / NA);
    atomicAdd(&rst[17 * batch_size + img_ith], Pxsy[img_ith * NA * Ng + ix * NA + ia] / (1 + float(ix) / Ng) * float(iy == 0) / NA);
    atomicAdd(&rst[18 * batch_size + img_ith], ix == 0? 0: float(iy == 0) * Pxsy[img_ith * NA * Ng + ix * NA + ia] / powf(ix, 2) / NA );

    if (ix == 0 and iy == 0)
   	{atomicAdd(&rst[19 * batch_size + img_ith], float(maxp[img_ith * NA + ia]) / (s[img_ith * NA + ia] + epsilon) / NA);}
    atomicAdd(&rst[20 * batch_size + img_ith], Pxay[img_ith * NA * Ng * 2 + ix * NA + ia] * (ix + 2) * float(iy == 0) / NA);
    atomicAdd(&rst[21 * batch_size + img_ith], -Pxay[img_ith * Ng * NA * 2 + ix * NA + ia] * log2f(Pxay[img_ith * Ng * NA * 2 + ix * NA + ia] + epsilon) * float(iy == 0) / NA);
    atomicAdd(&rst[22 * batch_size + img_ith], Pn[ip] * powf(float(ix + 1 - ux[img_ith * NA + ia]), 2) / NA);


}




/* Auto Correlation */
__global__ void f1_AutoCorrelation(float *rst, float *P, int Ng, int NA){

    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], P[ip] * (ix + 1) * (iy + 1));
	//*rst /= NA;
}

/* Joint Average */
__global__ void f2_JointAverage(float *rst, float *P, int Ng, int NA) {

    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

    atomicAdd(&rst[img_ith], P[ip] * float(ix + 1));
    //*rst /= NA;
}


/* CLuster Prominence */
__global__ void f3_ClusterProminence(float *rst, float *P, float *ux, float *uy, int Ng, int NA) {

    int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threads = blockDim.x * threadIdx.y + threadIdx.x;
    int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

    atomicAdd(&rst[img_ith], powf((float(ix + 1) + float(iy + 1) - ux[img_ith * NA + ia] - uy[img_ith * NA + ia]), 4) * P[ip]);
    //*rst /= NA;

}


/* ClusterShade */
__global__ void f4_ClusterShade(float *rst, float *P, float *ux, float *uy, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], powf((float(ix + 1) + float(iy + 1) - ux[img_ith * NA + ia] - uy[img_ith * NA + ia]), 3) * P[ip]);
	//*rst /= NA;
}


/* Cluster Tendency */
__global__ void f5_ClusterTendency(float *rst, float *P, float *ux, float *uy, int Ng, int NA) {

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], powf((float(ix + 1) + float(iy + 1) - ux[img_ith * NA + ia] - uy[img_ith * NA + ia]), 2) * P[ip]);
	//*rst /= NA;
}


/* Contrast */
__global__ void f6_Contrast(float *rst, float *P, int Ng, int NA) {

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], powf((ix - iy), 2) * P[ip]);
	//*rst /= NA;
}


/* Correlation */
__global__ void f7_Correlation(float *rst, float *P, float *ux, float *uy, float *Dx, float *Dy, int Ng, int NA, float epsilon) {

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], P[ip] * (ix + 1 - ux[img_ith * NA + ia]) * (iy + 1 - uy[img_ith * NA + ia]) /(sqrtf(Dx[img_ith * NA + ia] * Dy[img_ith * NA + ia]) + epsilon));
	//*rst /= NA;
}

/* Diffference Average */
__global__ void f8_DifferenceAverage(float *rst, float *DA, float *Pxsy, int Ng, int NA) {

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], float(ix) * Pxsy[img_ith * NA * Ng  + ix * NA + ia] * float(iy == 0));
	atomicAdd(&DA[img_ith * NA + ia], float(ix) * Pxsy[img_ith * NA * Ng + ix * NA + ia] * float(iy == 0));
	//*rst /= NA;
}


/* Differnence Entropy */
__global__ void f9_DifferenceEntropy(float *rst, float *Pxsy, float epsilon, int Ng, int NA) {

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], -Pxsy[img_ith * NA * Ng + ix * NA + ia] * log2f(Pxsy[img_ith * NA * Ng + ix * NA + ia] + epsilon) * float(iy == 0));
	//*rst /= NA;
}


/* Difference Variance */
__global__ void f10_DifferenceVariance(float *rst, float *Pxsy, float *DA, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	//atomicAdd(&DA[ia], float(ix) * Pxsy[ix * NA + ia] * float(iy == 0));
	atomicAdd(&rst[img_ith], powf(float(ix) - DA[img_ith * NA + ia], 2) * Pxsy[img_ith * Ng * NA + ix * NA + ia] * float(iy == 0));
	//*rst /= NA;
}

/* Joint Energy */
__global__ void f11_JointEnergy(float *rst, float *P, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], powf(P[ip], 2));
	//*rst /= NA;
}

/* Joint Entropy */
__global__ void f12_JointEntropy(float *rst, float *P, float epsilon, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], -P[ip] * log2f(P[ip] + epsilon));
	//*rst /= NA;
}

/* Information Measures of Correlation */
__global__ void f13_IMC1(float *rst, float *HXY, float *HXY1, float *HX, float *HY, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], float(ix == 0) * float(iy == 0) * (HXY[img_ith * NA + ia] - HXY1[img_ith * NA + ia]) / max(HX[img_ith * NA + ia], HY[img_ith * NA + ia]) );
	//*rst /= NA;
}

/* IMC2 */
__global__ void f14_IMC2(float *rst, float *HXY, float *HXY2, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], float(ix == 0) * float(iy == 0) * sqrtf(abs(1 - powf(M_E, -2 * (HXY2[img_ith * NA + ia] - HXY[img_ith * NA + ia])))));

	//*rst /= NA;
}

/* Inverse Difference Moment*/
__global__ void f15_IDM(float *rst, float *Pxsy, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], Pxsy[img_ith * NA * Ng + ix * NA + ia] / (1 + powf(ix, 2)) * float(iy == 0));
	//*rst /= 4;
}


/* Inverse Difference Moment Normalized*/

__global__ void f17_IDMN(float *rst, float *Pxsy, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], Pxsy[img_ith * NA * Ng + ix * NA + ia] / (1 + (powf(ix, 2)/powf(Ng, 2))) * float(iy == 0));
	//*rst /= NA;
}

/* Inverse Difference*/
__global__ void f18_ID(float *rst, float *Pxsy, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], Pxsy[img_ith * NA * Ng + ix * NA + ia] / float(1 + ix) * float(iy == 0));
	//*rst /= NA;
}

/* Inverse Difference Normalized*/
__global__ void f19_IDN(float *rst, float *Pxsy, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], Pxsy[img_ith * NA * Ng + ix * NA + ia] / (1 + float(ix) / Ng) * float(iy == 0));
	//*rst /= NA;
}

/* Inverse Variance*/
__global__ void f20_InverseVariance(float *rst, float *Pxsy, int Ng, int NA){

	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], ix == 0? 0: float(iy == 0) * Pxsy[img_ith * NA * Ng + ix * NA + ia] / powf(ix, 2) );
	//*rst /= NA;
}

/* Maximum Probability*/

__global__ void f21_MaximumProbability(float *rst, float *sum, int *maxp, int Ng, int NA, float epsilon){
	//float dst[4];
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

    if (ix == 0 and iy == 0)
	{atomicAdd(&rst[img_ith], float(maxp[img_ith * NA + ia]) / (sum[img_ith * NA + ia] + epsilon));}
    //printf("maxp: %f\n", maxp[0]);
    //atomicExch(&rst[img_ith], 0);
}

/* Sum Average*/
__global__ void f22_SumAverage(float *rst, float *Pxay, int Ng, int NA){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / (Ng / 2);
    iy = ipix / NA % (Ng / 2);
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], Pxay[img_ith * NA * Ng * 2 + ix * NA + ia] * (ix + 2) * float(iy == 0));
	//*rst /= NA;
}

/* Sum Entropy */
__global__ void f23_SumEntropy(float *rst, float *Pxay, float epsilon, int Ng, int NA){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

	int ix, iy, ia, ipix, img_ith;
	img_ith = ip / (Ng * Ng * NA);
	ipix = ip % (Ng * Ng * NA);
	ix = ipix / NA / (Ng / 2);
	iy = ipix / NA % (Ng / 2);
	ia = ipix % NA;

	atomicAdd(&rst[img_ith], -Pxay[img_ith * Ng * NA * 2 + ix * NA + ia] * log2f(Pxay[img_ith * Ng * NA * 2 + ix * NA + ia] + epsilon) * float(iy == 0));
	//*rst /= NA;
}

/*Sum of Squares*/
__global__ void f24_SumSquares(float *rst, float *P, float *ux, int Ng, int NA){
	int blocks = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int threads = blockDim.x * threadIdx.y + threadIdx.x;
	int ip = blocks * blockDim.x * blockDim.y + threads;

    int ix, iy, ia, ipix, img_ith;
    img_ith = ip / (Ng * Ng * NA);
    ipix = ip % (Ng * Ng * NA);
    ix = ipix / NA / Ng;
    iy = ipix / NA % Ng;
    ia = ipix % NA;

	atomicAdd(&rst[img_ith], P[ip] * powf(float(ix + 1 - ux[img_ith * NA + ia]), 2));
	//*rst /= NA;
}
