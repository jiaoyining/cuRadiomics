/*
#include "CVIPtexture.h"
//#include<stdio.h>

int main()
{	printf("start! \n");
	cudaDeviceProp devProp;
	int batch_size = 3; // the number of pictures

	char *imagename[] = {"//home//jyn//cuda-workspace//radiomics//image//image1",
						 "//home//jyn//cuda-workspace//radiomics//image//image2",
						 "//home//jyn//cuda-workspace//radiomics//image//image3"};
	char *maskname[] = {"//home//jyn//cuda-workspace//radiomics//image//mask1",
						"//home//jyn//cuda-workspace//radiomics//image//mask2",
						"//home//jyn//cuda-workspace//radiomics//image//mask3"};


	//char *imagename[] = {"//home//jyn//cuda-workspace//r//radiomics//image//image1"};
	//char *maskname[] = {"//home//jyn//cuda-workspace//r//radiomics//image//mask1"};


	//int *dev_image, *dev_mask, *dev_glcm, *glcm;

	PROPERTY *Property;
	TEXTURE *Texture;

	// loading images

	int *image, *mask;
	int size[] = {240, 240};

    image = (int *)malloc(sizeof(int) * size[0] * size[1] * batch_size);
    mask = (int *)malloc(sizeof(int) * size[0] * size[1] * batch_size);

	load_data(image, mask, imagename, maskname, size, batch_size);
	printf("loaded \n");
	// caluclating GLCM　Matrix
	START_TIMER(time)
	Texture = RadiomicsCalculator(image, mask, size, batch_size);
    printf("textures calculated \n");
    STOP_TIMER(time)
    PRINT_TIME(time)
    printf(" 1 ");
    //printf(Texture);

}
*/

