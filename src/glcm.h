/*
 * glcm.h
 *
 *  Created on: Jun 10, 2019
 *      Author: jyn
 */

#ifndef GLCM_H_
#define GLCM_H_

#include"headers.h"
#include"initialize.h"
typedef struct  {
	float *AutoCorrelation = NULL;           /*  (1) Angular Second Moment */
	float *JointAverage = NULL;      /*  (2) Contrast */
	float *ClusterProminence = NULL;   /*  (3) Correlation */
	float *ClusterShade = NULL;      /*  (4) Variance */
	float *ClusterTendency = NULL;		       /*  (5) Inverse Diffenence Moment */
	float *Contrast = NULL;	     /*  (6) Sum Average */
	float *Correlation = NULL;	     /*  (7) Sum Variance */
	float *DifferenceAverage = NULL;	 /*  (8) Sum Entropy */
	float *DifferenceEntropy = NULL;	     /*  (9) Entropy */
	float *DifferenceVariance = NULL;	     /* (10) Difference Variance */
	float *JointEnergy = NULL;	 /* (11) Diffenence Entropy */
	float *JointEntropy = NULL;	   /* (12) Measure of Correlation 1 */
	float *IMC1 = NULL;	   /* (13) Measure of Correlation 2 */
	float *IMC2 = NULL;
	float *IDM = NULL;
	float *IDMN = NULL;
	float *ID = NULL;
	float *IDN = NULL;
	float *InverseVariance = NULL;
	float *MaximumProbability = NULL;
	float *SumAverage = NULL;
	float *SumEntropy = NULL;
	float *SumSquares = NULL;
	/* (14) Maximal Correlation Coefficient */
} TEXTURE_glcm;

typedef struct  {
		int *P = NULL;
		float *s = NULL;
		float *Pn = NULL;
		float *Px = NULL;
		float *Py = NULL;
		float *ux = NULL;
		float *uy = NULL;
		float *Dx = NULL;
		float *Dy = NULL;
		float *Pxay = NULL;
		float *Pxsy = NULL;
		float *HX = NULL;
		float *HY = NULL;
		float *HXY = NULL;
		float *HXY1 = NULL;
		float *HXY2 = NULL;
		int *maxp = NULL;
		float *DA = NULL;
} PROPERTY_glcm;


int *Calculate_GLCM_rl(const int *image, int *size, int *stride, int *angles, int *Range, int MASK_VALUE, int bin_width, int Ng, int NA, int batch_size);
void Calculate_GLCM_Property(PROPERTY_glcm *Property_glcm, float Epsilon, int Ng, int NA, int batch_size);
void Calculate_GLCM_Texture_rl(PROPERTY_glcm *Property_glcm, float *texture_glcm, float Epsilon, int Ng, int NA, int n_batch);
__global__ void calculate_glcm_kernel(int *image, int *mask, int *glcm, int *dev_size, int *dev_stride, int *dev_angles, int dev_ng, int dev_na);
__global__ void calculate_glcm_kernel_rl(int *image, int *glcm, int *dev_size, int *dev_stride, int *dev_angles, int dev_ng, int dev_na);

// Properties of GLCM
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
						 );
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
						 );
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
						 );

__global__ void GLCM_sum(int *P, float *sum, int Ng, int NA);
__global__ void GLCM_Pn(int *P, float *Pn, float *sum, int Ng, int NA, float epsilon);
__global__ void GLCM_Px(float *P, float *Px, int Ng, int NA);
__global__ void GLCM_Py(float *P, float *Py, int Ng, int NA);
__global__ void GLCM_ux(float *P, float *ux, int Ng, int NA);
__global__ void GLCM_uy(float *P, float *uy, int Ng, int NA);
__global__ void GLCM_Dx(float *P,  float *ux, float *Dx, int Ng, int NA);
__global__ void GLCM_Dy(float *P,  float *uy, float *Dy, int Ng, int NA);
__global__ void GLCM_Pxay(float *P, float *Pxay, int Ng, int NA);
__global__ void GLCM_Pxsy(float *P, float *Pxsy, int Ng, int NA);
__global__ void GLCM_HX(float *Px, float *HX, float epsilon, int Ng, int NA);
__global__ void GLCM_HY(float *Py, float *HY, float epsilon, int Ng, int NA);
__global__ void GLCM_HXY(float *P, float *HXY, float epsilon, int Ng, int NA);
__global__ void GLCM_HXY1(float *P, float *Px, float *Py, float *HXY1, float epsilon, int Ng, int NA);
__global__ void GLCM_HXY2(float *P, float *Px, float *Py, float *HXY2, float epsilon, int Ng, int NA);
__global__ void GLCM_maxp(int *P, int *maxp, int Ng, int NA);
__global__ void GLCM_DA(float *DA, float *Pxsy, int Ng, int NA);


// GLCM Features
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
								   float epsilon);

__global__ void f1_AutoCorrelation(float *rst, float *P, int Ng, int NA);
__global__ void f2_JointAverage(float *rst, float *P, int Ng, int NA);
__global__ void f3_ClusterProminence(float *rst, float *P, float *ux, float *uy, int Ng, int NA);
__global__ void f4_ClusterShade(float *rst, float *P, float *ux, float *uy, int Ng, int NA);
__global__ void f5_ClusterTendency(float *rst, float *P, float *ux, float *uy, int Ng, int NA);
__global__ void f6_Contrast(float *rst, float *P, int Ng, int NA);
__global__ void f7_Correlation(float *rst, float *P, float *ux, float *uy, float *Dx, float *Dy, int Ng, int NA, float epsilon);
__global__ void f8_DifferenceAverage(float *rst, float *DA, float *Pxsy, int Ng, int NA);
__global__ void f9_DifferenceEntropy(float *rst, float *Pxsy, float epsilon, int Ng, int NA) ;
__global__ void f10_DifferenceVariance(float *rst, float *Pxsy, float *DA, int Ng, int NA) ;
__global__ void f11_JointEnergy(float *rst, float *P, int Ng, int NA);
__global__ void f12_JointEntropy(float *rst, float *P, float epsilon, int Ng, int NA);
__global__ void f13_IMC1(float *rst, float *HXY, float *HXY1, float *HX, float *HY, int Ng, int NA);
__global__ void f14_IMC2(float *rst, float *HXY, float *HXY2, int Ng, int NA);
__global__ void f15_IDM(float *rst, float *Pxsy, int Ng, int NA);
__global__ void f17_IDMN(float *rst, float *Pxsy, int Ng, int NA);
__global__ void f18_ID(float *rst, float *Pxsy, int Ng, int NA);
__global__ void f19_IDN(float *rst, float *Pxsy, int Ng, int NA);
__global__ void f20_InverseVariance(float *rst, float *Pxsy, int Ng, int NA);

__global__ void f21_MaximumProbability(float *rst, float *sum, int*maxp, int Ng, int NA,float epsilon);
__global__ void f22_SumAverage(float *rst, float *Pxay, int Ng, int NA);
__global__ void f23_SumEntropy(float *rst, float *Pxay, float epsilon, int Ng, int NA);
__global__ void f24_SumSquares(float *rst, float *P, float *ux, int Ng, int NA);


#endif /* GLCM_H_ */
