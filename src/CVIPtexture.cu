/*
** Author: James Darrell McCauley
**         Texas Agricultural Experiment Station
**         Department of Agricultural Engineering
**         Texas A&M University
**         College Station, Texas 77843-2117 USA
**
** Algorithms for calculating features (and some explanatory comments) are
** taken from:
**
**   Haralick, R.M., K. Shanmugam, and I. Dinstein. 1973. Textural features
**   for image classification.  IEEE Transactions on Systems, Man, and
**   Cybertinetics, SMC-3(6):610-621.
**
** Copyright (C) 1991 Texas Agricultural Experiment Station, employer for
** hire of James Darrell McCauley
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
** THE TEXAS AGRICULTURAL EXPERIMENT STATION (TAES) AND THE TEXAS A&M
** UNIVERSITY SYSTEM (TAMUS) MAKE NO EXPRESS OR IMPLIED WARRANTIES
** (INCLUDING BY WAY OF EXAMPLE, MERCHANTABILITY) WITH RESPECT TO ANY
** ITEM, AND SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL
** OR CONSEQUENTAL DAMAGES ARISING OUT OF THE POSESSION OR USE OF
** ANY SUCH ITEM. LICENSEE AND/OR USER AGREES TO INDEMNIFY AND HOLD
** TAES AND TAMUS HARMLESS FROM ANY CLAIMS ARISING OUT OF THE USE OR
** POSSESSION OF SUCH ITEMS.
**
** Modification History:
** Jun 24 91 - J. Michael Carstensen <jmc@imsor.dth.dk> supplied fix for
**             correlation function.
**
** Aug 7 96 - Wenxing Li: huge memory leaks are fixed.
**
** Nov 29 98 - M. Boland : Modified calculations to produce the same values**
**       as the kharalick routine in Khoros.  Some feature
**		 calculations below were wrong, others made different
**		 assumptions (the Haralick paper is not always explicit).
**
** Sep 1 04  - T. Macura : Modified for inclusion in OME by removing all
**       dependencies on pgm/ppm. Refactored co-occurence matrix calculations
**       to be more modular to facilitate future support of angles other than
**       0, 45, 90 and 135. Limits on co-occurence matrix direction and
**       quantisation level of image removed.
**
**       Previously when quantisation level was limited to 255, a statically
**       allocated array served as pixel_intensity -> CoOcMat array index
**       lut. lut was sequentially searched. Now lut is hash table provided by
**       mapkit.c. It can dynamically grow from initialize size of 1024. This is
**       expected to improve performance drastically for larger images and images
**       with higher quantisations.
**
** Sep 20 04 - T. Macura : ++ precision, all floats replaced with doubles
** Jan 12 07 - T. Macura : removing hash tables, going back to LUT because
**                         we aren't actually using quantizations higher than 255
**                         and the hash-tables have memory-leaks.
**
*/

#define _USE_MATH_DEFINES
#include "CVIPtexture.h"

void  RadiomicsCalculator_rl(const int *image, float *texture, const int *SET, int batch_size, int size0, int size1){

	int Range[] = {SET[0], SET[1]};
	//int Range[] = {0, 255};
	int BW = BIN_WIDTH;
	int Ng = (SET[1] - SET[0] + 1) / BW;

	int choose[] = {SET[2], SET[3]};
	int MASK_VALUE = SET[4];

	float Epsilon = EPSILON;
	int size[] = {size0, size1};
	int stride[] = {size1, 1};
	int n_batch = batch_size;
	//int MASK_VALUE = mask_value[0];


	// Constant Values concerning Directions of GLCM
	int NA = 4;
	int angles[] = {
		1, 1,
		1, 0,
		1, -1,
		0, 1};

	//printf("%d， %d, %d, %d, %d, %d", SET[0], SET[1], SET[2], SET[3], SET[4], SET[5]);
	if (choose[0] == 1)
	{
		PROPERTY_glcm Property_glcm;

	    (&Property_glcm)->P = Calculate_GLCM_rl(image, size, stride, angles, Range, MASK_VALUE, BW, Ng, NA, n_batch);
		//(&Property_fo)->P = Calculate_firstorder_rl(image, size, stride, bin, Binwidth, n_batch);
		//printf("glcm caculated! \n");

		//printf("memory allocated! \n");
		Calculate_GLCM_Property(&Property_glcm, Epsilon, Ng, NA, n_batch);
		//Calculate_firstorder_Property(&Property_fo, Epsilon, bin, Binwidth, Ng,  n_batch);
		//printf("property_glcm calculated! \n");

		Calculate_GLCM_Texture_rl(&Property_glcm, texture, Epsilon, Ng, NA, n_batch);
		//Calculate_firstorder_Texture_rl(&Property_fo, &texture[23 * n_batch], Epsilon, bin, Ng, Binwidth, n_batch);
		cudaFree(Property_glcm.P);
		cudaFree(Property_glcm.Pn);

		cudaFree(Property_glcm.Px);
		cudaFree(Property_glcm.Py);
		cudaFree(Property_glcm.s);
		cudaFree(Property_glcm.ux);
		cudaFree(Property_glcm.uy);
		cudaFree(Property_glcm.Dx);
		cudaFree(Property_glcm.Dy);
		cudaFree(Property_glcm.Pxay);

		cudaFree(Property_glcm.Pxsy);
		cudaFree(Property_glcm.HX);
		cudaFree(Property_glcm.HY);
		cudaFree(Property_glcm.HXY);
		cudaFree(Property_glcm.HXY1);
		cudaFree(Property_glcm.HXY2);
		cudaFree(Property_glcm.DA);
		cudaFree(Property_glcm.maxp);
	}

	if (choose[1] == 1)
	{

	PROPERTY_fo Property_fo;
	(&Property_fo)->P = Calculate_firstorder_rl(image, Range, MASK_VALUE, size, stride, Ng, BW, n_batch);
	Calculate_firstorder_Property(&Property_fo, Epsilon, BW, Ng, n_batch);
	Calculate_firstorder_Texture_rl(&Property_fo, &texture[23 * n_batch * choose[0]], Epsilon, Ng, BW, n_batch);

	cudaFree(Property_fo.P);
	cudaFree(Property_fo.Np);
	cudaFree(Property_fo.Pn);

	cudaFree(Property_fo.pn);
	cudaFree(Property_fo.Pf);
	cudaFree(Property_fo.PF);
	cudaFree(Property_fo.P25);
	cudaFree(Property_fo.P50);
	cudaFree(Property_fo.P75);
	cudaFree(Property_fo.P90);
	cudaFree(Property_fo.P10);

	cudaFree(Property_fo.Pmin);
	cudaFree(Property_fo.Pmax);
	cudaFree(Property_fo.Pm);
	cudaFree(Property_fo.Pv);
	cudaFree(Property_fo.N1090);
	cudaFree(Property_fo.mP1090);
	}

	//printf("Texture_firstorder Calculated! \n");

	//texture_glcm = out_texture_glcm;

	//Property_glcm.~PROPERTY_glcm();
	//cudaFree(glcm);

	//FreeProperty_glcm(Property_glcm);
	//cudaFree(out_texture_glcm);


	//free(Property_glcm);
	//free(size);
	//free(angles);
	//free(stride);


	//cudaFree(maxp);



/*
	cudaFree(Property_glcm->P);
	cudaFree(Property_glcm->Pn);

	cudaFree(Property_glcm->Px);
	cudaFree(Property_glcm->Py);
	cudaFree(Property_glcm->s);
	cudaFree(Property_glcm->ux);
	cudaFree(Property_glcm->uy);
	cudaFree(Property_glcm->Dx);
	cudaFree(Property_glcm->Dy);
	cudaFree(Property_glcm->Pxay);

	cudaFree(Property_glcm->Pxsy);
	cudaFree(Property_glcm->HX);
	cudaFree(Property_glcm->HY);
	cudaFree(Property_glcm->HXY);
	cudaFree(Property_glcm->HXY1);
	cudaFree(Property_glcm->HXY2);
	cudaFree(Property_glcm->maxp);
	*/





	//printf("deleted! \n");
	//delete []Property_glcm;

	//free(Property_glcm);
	//free(glcm);

}















