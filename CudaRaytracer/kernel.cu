#include <iostream>
#include "constants.h"

#include "ray.h"
#include "sphere.h"
#include "mathematics.h"

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

	geometry::Ray ray(make_float3(X, Y, -1000.f), make_float3(0.f, 0.f, 1.f));
	geometry::Sphere sphere(make_float3(300.f, 300.f, 0.f), 150.f);

	float3 dist = sphere.origin - ray.origin;
	double B = math::dot(ray.direction, dist);
	double D = B*B - math::dot(dist, dist) + sphere.radius * sphere.radius; 
    
    double t0 = B - sqrt(D); 
    double t1 = B + sqrt(D);
    
    if ((t0 > 0.1f) || (t1 > 0.1f)) 
    {
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
