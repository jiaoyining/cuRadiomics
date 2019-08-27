/*
 * initialize.h
 *
 *  Created on: Jun 10, 2019
 *      Author: jyn
 */

#ifndef INITIALIZE_H_
#define INITIALIZE_H_

#include"headers.h"

__global__ void initialize(int *glcm);
__global__ void initialize_tex(float *texture);
__global__ void initialize_mtex(float *texture);
__global__ void Preprocessing_image_GLCM(int *dev_image, const int *image, int Min_V, int Max_V, int bin_width, int MASK_V);
__global__ void Preprocessing_image_firstorder(int *dev_image, const int *image, int Min_V, int Max_V, int MASK_V);



#endif /* INITIALIZE_H_ */

