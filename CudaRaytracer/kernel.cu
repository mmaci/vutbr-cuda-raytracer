#include <iostream>
#include "constants.h"

#include "ray.h"
#include "sphere.h"
#include "plane.h"
#include "mathematics.h"
#include "camera.h"

using namespace CUDA;

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
	
	float x = (2.f*X/WINDOW_WIDTH - 1.f);
	float y = (2.f*Y/WINDOW_HEIGHT - 1.f);

	Camera cam;	
	cam.lookAt(make_float3(2, 3, -7),  // eye
             make_float3(5, 0, 1),   // target
             make_float3(0, 1, 0),   // sky
             30, (float)WINDOW_WIDTH/WINDOW_HEIGHT);
	
	Sphere s(make_float3(-1.7f, 4.f, 0.f), 1.6f);		
	Plane p(make_float3(0, 0, 1), make_float3(0, 0, 15));
	
	if (s.intersect(cam.getRay(x, y))) {		
		data[WINDOW_WIDTH * Y + X].x = float(X) / float(WINDOW_WIDTH) * 255.f;
		data[WINDOW_WIDTH * Y + X].y = 0;
		data[WINDOW_WIDTH * Y + X].z = float(Y) / float(WINDOW_WIDTH) * 255.f;
		data[WINDOW_WIDTH * Y + X].w = 0;
				
	}
	else 
	{
		data[WINDOW_WIDTH * Y + X].x = 0;
		data[WINDOW_WIDTH * Y + X].y = 0;
		data[WINDOW_WIDTH * Y + X].z = 0;
		data[WINDOW_WIDTH * Y + X].w = 0;
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
	dim3 threadsPerBlock(8, 8, 1); // 64 threads ~ 8*8
	dim3 numBlocks(WINDOW_WIDTH / threadsPerBlock.x, WINDOW_HEIGHT / threadsPerBlock.y);

	RTKernel<<<numBlocks, threadsPerBlock>>>(data, imageWidth, imageHeight);
   	
	cudaThreadSynchronize();
	checkCUDAError();
}
