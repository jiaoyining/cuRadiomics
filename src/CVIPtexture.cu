

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















