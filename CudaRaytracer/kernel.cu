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
	uint32 i = blockIdx.x * blockDim.x + threadIdx.x;  
 
	if (i < WINDOW_SIZE)
	{
		// some random stuff
		// here the raytracer will shoot the rays

		uint8 r = (i * width) & 0xff;
		uint8 g = (i * height) & 0xff;
		uint8 b = (i * (width + height)) & 0xff;
     		
		data[i].w = 0;
		data[i].x = r;
		data[i].y = g;
		data[i].z = b;
	}
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
	uint32 totalThreads = imageHeight * imageWidth;
	uint32 numBlocks = totalThreads / NUM_THREADS;
	numBlocks += ((totalThreads % NUM_THREADS) > 0) ? 1 : 0;
 
	RTKernel<<<numBlocks, NUM_THREADS>>>(data, imageWidth, imageHeight);
   	
	cudaThreadSynchronize();
	checkCUDAError();
}
