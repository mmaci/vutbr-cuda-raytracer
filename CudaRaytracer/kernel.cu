#include <iostream>
#include "constants.h"

/**
 * Checks for error and if found writes to cerr and exits program. 
 */
void checkCUDAError()
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err)
	{
		std::cerr << "Cuda error: " << cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
}

/**
 * CUDA kernel
 *
 * @param uchar* data
 * @param uint32 width
 * @param uint32 height
 * @param float time
 */
__global__ void RTKernel(uchar4* data, uint32 width, uint32 height)
{
	uint32 X = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32 Y = (blockIdx.y * blockDim.y) + threadIdx.y;

	data[WINDOW_WIDTH * Y + X].x = float(X) / float(WINDOW_WIDTH) * 255.f;
	data[WINDOW_WIDTH * Y + X].y = float(Y) / float(WINDOW_HEIGHT) * 255.f;
	data[WINDOW_WIDTH * Y + X].z = 0;
	data[WINDOW_WIDTH * Y + X].w = 0;
}
 
/**
 * Wrapper for the CUDA kernel
 *
 * @param uchar* data
 * @param uint32 width
 * @param uint32 height
 * @param float time
 */
extern "C" void launchRTKernel(uchar4* data, uint32 imageWidth, uint32 imageHeight)
{   	
	dim3 threadsPerBlock(8, 8, 1); // 64 threads ~ 8*8
	dim3 numBlocks(WINDOW_WIDTH / threadsPerBlock.x, WINDOW_HEIGHT / threadsPerBlock.y);

	RTKernel<<<numBlocks, threadsPerBlock>>>(data, imageWidth, imageHeight);
   	
	cudaThreadSynchronize();
	checkCUDAError();
}
