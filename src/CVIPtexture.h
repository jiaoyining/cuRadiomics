
#ifndef _CVIP_texture
#define _CVIP_texture

#include "parameters.h"
#include "initialize.h"
#include "firstorder.h"
#include "glcm.h"
#include "headers.h"

//#include<glszm.h>

/* 
[0] -> 0 degree, 
[1] -> 45 degree, 
[2] -> 90 degree, 
[3] -> 135 degree,
[4] -> average, 
[5] -> range (max - min) 
*/


void RadiomicsCalculator_rl(const int *image, float *texture, const int *SET, int batch_size, int size0, int size1);

//__global__ void Calculate_GLCM_Property_kernel(PROPERTY *Property, int *P_matrix, int NA, int Ng, float Epsilon);

#endif
