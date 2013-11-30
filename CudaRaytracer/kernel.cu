#include <iostream>
#include "constants.h"

#include "ray.h"
#include "sphere.h"
#include "mathematics.h"
#include "camera.h"
#include "plane.h"
#include "light.h"
#include "scene.h"

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

//__device__ Color TraceRay(const Ray &ray, Scene &scene, PointLight &light, int recursion){
//	if (scene.Intersect(ray)) {
	
//		return scene.interSColor;
//	} else
//	{
//		return Color(0, 0, 0);
//	}
//}
	


__device__ Color TraceRay(const Ray &ray,  Plane &p,  Sphere &s, PointLight &light, int recursion)
{
	float st = s.intersect(ray);
	float pt = p.intersect(ray);
	float3 point;
	float3 normal;
	
	if ((pt==0.f) && (st == 0.f)){ //miss
		return Color(0,0,0);
	} else if (pt > st) //plane hit
	{
		point = ray.getPoint(pt);
		normal = p.normal;
		return p.color;

	} else if (st >= pt) //sphere hit
	{
		point = ray.getPoint(st);
		normal = s.getNormal(point);
		return s.color;
	};

	return Color();
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
	Ray ray = cam.getRay(x,y);
	Scene scene;

	//scene.Add(Sphere(make_float3(8.f, 4.f, 0.f), 2.f,Color(255.f,0,0)));
	//scene.Add(Plane(make_float3(10, 50, 100), make_float3(5.f, 0.f, 0.f),Color(0,0,255.f)));
	Sphere s(make_float3(8.f, 4.f, 0.f), 2.f,Color(255.f,0,0));		
	Plane p(make_float3(10, 50, 100), make_float3(5.f, 0.f, 0.f),Color(0,0,255.f));
    //scene.Add(s);
	//scene.Add(p);

	PointLight l(make_float3(8.f, 10.f, 2.f),Color(0,255.f,0));
	//Color c = TraceRay(ray,scene,l,15);
	Color c = TraceRay(ray,p,s,l,15);
	data[WINDOW_WIDTH * Y + X].x = c.red;
	data[WINDOW_WIDTH * Y + X].y = c.green;
	data[WINDOW_WIDTH * Y + X].z = c.blue;
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
