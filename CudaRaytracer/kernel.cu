#include <iostream>
#include "constants.h"

#include "ray.h"
#include "sphere.h"
#include "mathematics.h"
#include "camera.h"
#include "plane.h"

#include "scene.h"
#include "phong.h"

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


__device__ HitInfo intersectRayWithScene(Ray const& ray,  Plane* planes,  Sphere* spheres, SceneStats* sceneStats)
{
	HitInfo hitInfo, hit;
	
	float st = FLT_MAX;
	float pt = FLT_MAX;
	int maxPi = NO_HIT;
	int maxSi = NO_HIT;
		
	uint32 i;
	// SPHERES
	for (i = 0; i < sceneStats->sphereCount; ++i)
	{
		hit = spheres[i].intersect(ray);
		if (hit.hit)
		{
			if (st > hit.t)
			{
				st = hit.t;
				maxSi = i;
			}	
		}
	}
	for (i = 0; i < sceneStats->planeCount; ++i){
		hit = planes[i].intersect(ray);
		if (hit.hit){
			if (pt > hit.t){
				pt = hit.t;
				maxPi = i;
			}			
		}
	}
	// miss
	if ((maxPi<0) && (maxSi < 0))
	{ 		
		hitInfo.hit = false;
		return hitInfo;		
	}
	// PLANE hit
	else if (pt < st)
	{
		hitInfo.t = pt;
		hitInfo.point = ray.getPoint(pt);
		hitInfo.normal = planes[maxPi].normal;		
		hitInfo.phongInfo = planes[maxPi].phong;
	}
	// SPHERE hit
	else if (st < pt)
	{
		hitInfo.t = st;
		hitInfo.point = ray.getPoint(st);
		hitInfo.normal = spheres[maxSi].getNormal(hitInfo.point);		
		hitInfo.phongInfo = spheres[maxSi].phong;
	}
	hitInfo.hit = true;
	return hitInfo;	
}


__device__ Color TraceRay(const Ray &ray,  Plane* planes, Sphere* spheres, PointLight* lights, SceneStats* sceneStats, int recursion)
{
	Color color;

	HitInfo hitInfo = intersectRayWithScene(ray, planes, spheres, sceneStats);
	if (hitInfo.hit)
	{

		for (uint32 i = 0; i < sceneStats->lightCount; ++i)
		{	
			PhongInfo phongInfo = hitInfo.phongInfo;
			const float3 hitPoint = CUDA::float3_add(ray.origin, CUDA::float3_mult(hitInfo.t, ray.direction));		
			const float3 hitNormal = hitInfo.normal;
			const float3 lightPos = lights[i].position;		
			const float3 shadowDir = CUDA::normalize(CUDA::float3_sub(lightPos, hitPoint));

			float intensity = CUDA::dot(hitNormal, shadowDir);

			if (intensity > 0.f) { // only if there is enought light
				Ray lightRay = Ray(lights[i].position, CUDA::float3_sub(hitPoint, lightPos));

				HitInfo shadowHit = intersectRayWithScene(lightRay, planes, spheres, sceneStats);

				if ((shadowHit.hit) && (shadowHit.t < hitInfo.t - FLT_EPSILON)) {
					color.accumulate(CUDA::mult(phongInfo.diffuse, lights[i].color), intensity);
				}
			}
		}
						
			
			/*
			if (phongInfo.shininess != 0.f) {
				float3 rlv = float3_sub(CUDA::cross(float3_mult(2.f, CUDA::cross(lv,hitInfo.normal)), hitInfo.normal) , lv);
				float3 vv = ray.direction;
				vv = CUDA::normalize(vv);
				float specular = -1.f * CUDA::dot(rlv, vv);
				if (specular > 0) {
					color.accumulate(mult(hitInfo.phongInfo.specular, light.color), pow(specular, hitInfo.phongInfo.shininess));
				}
			}*/		

		//reflected ray
		/*if ((hitInfo.phongInfo.reflectance>0) && (recursion > 0)) {
			Ray rray(hitInfo.point, float3_sub(ray.direction, CUDA::cross(float3_mult(2,CUDA::cross(ray.direction,hitInfo.normal)) ,hitInfo.normal)));
			rray.ShiftStart(1e-5);

			//Color rcolor = TraceRay(rray, p,s, light,sceneStats, recursion-1);
			//        color *= 1-phong.GetReflectance();
			//color.accumulate(rcolor, hitInfo.phongInfo.reflectance);
		}*/		
	}
	
	return color;
}	

/**
* CUDA kernel
*
* @param uchar* data
* @param uint32 width
* @param uint32 height
* @param float time
*/
__global__ void RTKernel(uchar3* data, uint32 width, uint32 height, Sphere* spheres, Plane* planes, PointLight* lights, SceneStats* sceneStats, Camera* camera)
{
	uint32 X = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32 Y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float x = (2.f*X/WINDOW_WIDTH - 1.f);
	float y = (2.f*Y/WINDOW_HEIGHT - 1.f);

	//float dx = 2.0/width;
	//float dy = 2.0/height;

	Ray ray = camera->getRay(x,y);
	
	//Color c = TraceRay(ray,scene,l,15);
	Color c = TraceRay(ray, planes, spheres, lights, sceneStats, 5);


	//tohle pak asi pryc
	/*Color cl(data[WINDOW_WIDTH * Y + X-1].x/255.f,data[WINDOW_WIDTH * Y + X-1].y/255.f,data[WINDOW_WIDTH * Y + X-1].z/255.f);
	Color ct(data[(WINDOW_WIDTH-1) * Y + X-1].x/255.f,data[(WINDOW_WIDTH-1)  * Y + X-1].y/255.f,data[(WINDOW_WIDTH-1)  * Y + X-1].z/255.f);
	const float thresh = -5.0001;
	if (diff(c, cl) > thresh || diff(c, ct) > thresh) {
	//          cout << diff(c, cl) << "  " << diff(c, ct) << endl;
	Color cc;      
	ray = camera->getRay(x - dx/3, y - dy/3);
	cc = TraceRay(ray,planes, spheres, l,sceneStats, 15);
	c.accumulate(cc, 0.6);
	ray = camera->getRay(x - dx/3, y + dy/3);
	cc = TraceRay(ray, planes, spheres, l,sceneStats, 15);
	c.accumulate(cc, 0.6);
	ray = camera->getRay(x + dx/3, y - dy/3);
	cc = TraceRay(ray,planes, spheres, l,sceneStats, 15);
	c.accumulate(cc, 0.6);
	ray = camera->getRay(x + dx/3, y + dy/3);
	cc = TraceRay(ray, planes, spheres, l,sceneStats, 15);
	c.accumulate(cc, 0.6);

	c *= 1/(1 + 4*0.6);
	}*/

	data[WINDOW_WIDTH * Y + X].x = min(c.red* 255.f, 255.f);
	data[WINDOW_WIDTH * Y + X].y = min(c.green* 255.f, 255.f);
	data[WINDOW_WIDTH * Y + X].z = min(c.blue* 255.f, 255.f);	
	/*data[WINDOW_WIDTH * Y + X].x = c.red*255.f;
	data[WINDOW_WIDTH * Y + X].y = c.green*255.f ;
	data[WINDOW_WIDTH * Y + X].z = c.blue*255.f ;
	data[WINDOW_WIDTH * Y + X].w = 0;*/

}


/**
* Wrapper for the CUDA kernel
*
* @param uchar* data
* @param uint32 width
* @param uint32 height
* @param float time
*/
extern "C" void launchRTKernel(uchar3* data, uint32 imageWidth, uint32 imageHeight, Sphere* spheres, Plane* planes, PointLight* lights, SceneStats* sceneStats, Camera* camera)
{   	
	dim3 threadsPerBlock(8, 8, 1); // 64 threads ~ 8*8
	dim3 numBlocks(WINDOW_WIDTH / threadsPerBlock.x, WINDOW_HEIGHT / threadsPerBlock.y);

	RTKernel<<<numBlocks, threadsPerBlock>>>(data, imageWidth, imageHeight, spheres, planes, lights, sceneStats, camera);

	cudaThreadSynchronize();
	checkCUDAError();
}
