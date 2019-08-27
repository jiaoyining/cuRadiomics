/*
 * cuda_op_kernel.cu.cc
 *
 *  Created on: Mar 12, 2019
 *      Author: jyn
 */
//#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "CVIPtexture.h"
#include <time.h>

using namespace tensorflow;
//using namespace CVIP_texture;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("Radiomics")
.Input("image: int32")
.Input("set: int32")
//.Input("BinWidth: int32")
//.Input("chosen: int32")
//.Input("MASK_VALUE: int32")
//.Output("time: float32")
.Output("texture: float32")
.Doc(R"doc(  
GLCM radiomics feature  
)doc");

class RadiomicsOp : public OpKernel {
public:
	explicit RadiomicsOp(OpKernelConstruction* context) : OpKernel(context){};

	void Compute(OpKernelContext* context) override {
		const Tensor& input_tensor = context->input(0);
		//const Tensor& mask_tensor = context-> input(1);
		const Tensor& Set_tensor = context->input(1);
		//const Tensor& BinWidth_tensor = context->input(3);
		//const Tensor& choose_tensor = context->input(4);
		//const Tensor& mv_tensor = context->input(5);

		auto input = input_tensor.flat<int32>();
		//auto mask = mask_tensor.flat<int32>();
		auto set = Set_tensor.flat<int32>();
		//auto BW = BinWidth_tensor.flat<int32>();
		//auto choose = choose_tensor.flat<int32>();
		//auto mask_value = choose_tensor.flat<int32>();

		int batch_size = input_tensor.shape().dim_size(0);
		int size0 = input_tensor.shape().dim_size(1);
		int size1 = input_tensor.shape().dim_size(2);

		//printf("%d \n", batch_size);

		//int size[] = {224, 224};
		//const int N = input.size() / batch_size;

		Tensor* output_tensor = NULL;
		int num_features = 0;

		//reading what is needed to be calculated, GLCM or fistorder
		//according to the chosen vector, distribute the shape of the output tensor
		//int *chosen = (int*)malloc(sizeof(int) * 2);
		//cudaMemcpy(chosen, choose.data(), sizeof(int) * 2, cudaMemcpyDeviceToHost);
		//printf("%d, %d", chosen[0], chosen[1]);
		int *SET = (int*)malloc(sizeof(int) * 5);
		cudaMemcpy(SET, set.data(), sizeof(int) * 5, cudaMemcpyDeviceToHost);
		const int SET0[] = {SET[0], SET[1], SET[2], SET[3], SET[4]};
		//printf("%d, %d \n", SET[3], SET[4]);
		if(SET[2] == 1)
			{
			num_features += 23;
			}
		if(SET[3] == 1)
		{
			num_features += 18;
		}

		TensorShape shape_of_out = TensorShape({num_features * batch_size});
		// a.AddDim( num_of_features * batch_size);//= num_of_features * batch_size;
		OP_REQUIRES_OK(context, context->allocate_output(0, shape_of_out, &output_tensor));
		auto output = output_tensor->flat<float>();

		/*
		// reading Ng, in order to distribute the GPU memory for Radiomics matrix
		int *Range = (int*)malloc(sizeof(int) * 2);
		cudaMemcpy(Range, range.data(), sizeof(int) * 2, cudaMemcpyDeviceToHost);
		// reading BinWidth, in order to distribute the GPU memory for Radiomics matrix
		int *BinWidth = (int*)malloc(sizeof(int) * 1);
		cudaMemcpy(BinWidth, BW.data(), sizeof(int) * 1, cudaMemcpyDeviceToHost);
		// reading , in order to distribute the GPU memory for Radiomics matrix
		int *MASK_VALUE = (int*)malloc(sizeof(int) * 1);
		cudaMemcpy(MASK_VALUE, mask_value.data(), sizeof(int) * 1, cudaMemcpyDeviceToHost);
		*/
		// Feature Extraction
	    RadiomicsCalculator_rl(input.data(), output.data(), SET0, batch_size, size0, size1);
	   // cudaDeviceSynchronize();
	   // end = clock();
	   // printf("TIME: %f\n", float(end-start)/ CLOCKS_PER_SEC);
	    		//return ;

	    };
};



REGISTER_KERNEL_BUILDER(Name("Radiomics").Device(DEVICE_GPU), RadiomicsOp);

//#endif


