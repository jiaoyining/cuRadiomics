/*
 * firstorder.h
 *
 *  Created on: Jun 10, 2019
 *      Author: jyn
 */

#ifndef FIRSTORDER_H_
#define FIRSTORDER_H_

#include"headers.h"
#include"initialize.h"

typedef struct  {
	float *Energy = NULL;           /*  (1) Angular Second Moment */
	float *TotalEnergy = NULL;      /*  (2) Contrast */
	float *Entropy = NULL;   /*  (3) Correlation */
	float *Minimum = NULL;      /*  (4) Variance */
	float *TenthPercentile = NULL;		       /*  (5) Inverse Diffenence Moment */
	float *NintiethPercentile = NULL;	     /*  (6) Sum Average */
	float *Maximum = NULL;	     /*  (7) Sum Variance */
	float *Mean = NULL;	 /*  (8) Sum Entropy */
	float *Median = NULL;	     /*  (9) Entropy */
	float *InterquartileRange = NULL;	     /* (10) Difference Variance */
	float *Range = NULL;	 /* (11) Diffenence Entropy */
	float *MAD = NULL;
	float *rMAD = NULL; /* (12) Measure of Correlation 1 */
	float *RMS= NULL;	   /* (13) Measure of Correlation 2 */
	float *StandardDeviation = NULL;
	float *Skewness = NULL;
	float *Kurtosis = NULL;
	float *Variance = NULL;
	float *Uniformity = NULL;

} TEXTURE_fo;


typedef struct  {
		int *P = NULL;
		int *PF = NULL;
		float *pn = NULL;
		float *Pf = NULL;
		float *Np = NULL;
		float *Pn = NULL;
		int *P25 = NULL;
		int *P50 = NULL;
		int *P75 = NULL;
		int *P10 = NULL;
		int *P90 = NULL;
		float *mP1090 = NULL;
		int *N1090 = NULL;
		int *Pmin = NULL;
	    int *Pmax = NULL;
		float *Pm = NULL;
		float *Pv = NULL;

} PROPERTY_fo;



int *Calculate_firstorder_rl(const int *image, int *Range, int MASK_VALUE, int *size, int *stride, int Ng, int bin_width, int batch_size);
void Calculate_firstorder_Property(PROPERTY_fo *Property_fo, float Epsilon,  int bin_width, int Ng, int batch_size);
void Calculate_firstorder_Texture_rl(PROPERTY_fo *Property_fo, float *texture_fo, float Epsilon, int Ng, int bin_width, int batch_size);
__global__ void calculate_firstorder_kernel_rl(int *image, int *P, int *dev_size, int *dev_stride, int dev_bin);


// Properties of  firstorder

__global__ void firstorder_Np(int *P, float *Np, int dev_bin);
__global__ void firstorder_Pn(float *Pn, int *P, float *Np, int dev_bin);
__global__ void firstorder_pn(float *pn, float *Pn, int dev_ng, int dev_bin, int bin_width);
__global__ void firstorder_PF(int *PF, int *P, int dev_bin);
__global__ void firstorder_Pf(float *Pf, int *PF, float *Np, int dev_bin);
__global__ void firstorder_P25(int *P25, float *Pf, int dev_bin);
__global__ void firstorder_P50(int *P50, float *Pf, int dev_bin);
__global__ void firstorder_P75(int *P75, float *Pf, int dev_bin);
__global__ void firstorder_P10(int *P10, float *Pf, int dev_bin);
__global__ void firstorder_P90(int *P90, float *Pf, int dev_bin);
__global__ void firstorder_N1090(int *N1090, int *P, int *P10, int *P90, int dev_bin);
__global__ void firstorder_mP1090(float *mP1090, int *N1090, float *Pn, float *Np, int *P10, int *P90, int dev_bin);
__global__ void firstorder_Pmin(int *Pmin, int *PF, int dev_bin);
__global__ void firstorder_Pmax(int *Pmax, int *PF, int dev_bin);
__global__ void firstorder_Pm(float *Pm, float *Pn, int dev_bin);
__global__ void firstorder_Pv(float *Pv, float *Pm, float *Pn, int dev_bin);


// Features of first order
__global__ void f1_Energy(float *rst, float *Pn, float *Np, int dev_bin);
__global__ void f3_Entropy(float *rst, float *pn, float *Np, int dev_bin, int bin_width, float Epsilon);
__global__ void f4_Minimum(float *rst, int *Pmin, int dev_bin);
__global__ void f5_TenthPercentile(float *rst, int *P10, int dev_bin);
__global__ void f6_NinetiePercentile(float *rst, int *P90, int dev_bin);
__global__ void f7_Maximum(float *rst, int *Pmax, int dev_bin);
__global__ void f8_Mean(float *rst, float *Pm, int dev_bin);
__global__ void f9_Median(float *rst, int *P50, int dev_bin);
__global__ void f10_InterquartileRange(float *rst, int *P25, int *P75, int dev_bin);
__global__ void f11_Range(float *rst, int *Pmin,int *Pmax, int dev_bin);
__global__ void f12_MAD(float *rst, float *Pn, float *Pm, float *Np, int dev_bin);
__global__ void f13_rMAD(float *rst, int *N1090, float *mP1090, int *P10, int *P90, float *Pn, float *Np, int dev_bin);
__global__ void f14_RMS(float *rst, float *Energy, float *Np, int dev_bin);
__global__ void f15_StandardDeviation(float *rst, float *Pv, int dev_bin);
__global__ void f16_Skewness(float *rst, float *Pm, float *Pn, float *Pv, int dev_bin, float Epsilon);
__global__ void f17_Kurtosis(float *rst, float *Pm, float *Pn, float *Pv, int dev_bin, float Epsilon);
__global__ void f19_Uniformity(float *rst, float *pn, int dev_bin, int bin_width);
__global__ void f18_Variance(float *rst, float *Pv, int dev_bin);
__global__ void f20_Volume(float *rst, float *Np, int dev_bin);








#endif /* FIRSTORDER_H_ */
