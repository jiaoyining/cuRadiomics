/***************************************************************************
* ======================================================================
* Computer Vision/Image Processing Tool Project - Dr. Scott Umbaugh SIUE
* ======================================================================
*
*             File Name: CVIPtexture.h
*           Description: contains function prototypes, type names, constants,
*			 etc. related to libdataserv (Data Services Toolkit.)
*         Related Files: Imakefile, cvip_pgmtexture.c
*   Initial Coding Date: 6/19/96
*           Portability: Standard (ANSI) C
*             Credit(s): Steve Costello
*                        Southern Illinois University @ Edwardsville
*
** Copyright (C) 1993 SIUE - by Gregory Hance.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
****************************************************************************/
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
