#include <iostream>
#include "constants.h"

#include "ray.h"
#include "sphere.h"
#include "mathematics.h"
#include "camera.h"
#include "plane.h"
#include "light.h"
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


__device__ HitInfo Intersect(const Ray &ray,  Plane* p,  Sphere* s, SceneStats* sceneStats){

	HitInfo hitInfo;
	int i;
	float st,pt;
	st = 1e999;
	pt = 1e999;
	int maxPi = -1;
	int maxSi = -1;
	//float tmp;
	HitInfo hit;
	for (i=0;i<(sceneStats->SphereCount);i++){
		hit = s[i].intersect(ray);
		if (hit.hit ){
			if (st > hit.t){
				st = hit.t;
				maxSi =i;
			}	
		}
	}
	for (i=0;i<(sceneStats->PlaneCount);i++){
		hit = p[i].intersect(ray);
		if (hit.hit){
			if (pt > hit.t){
				pt = hit.t;
				maxPi = i;
			}			
		}
	}
	if ((maxPi<0) && (maxSi < 0))//miss
	{ 
		//t = 0.f;
		hitInfo.hit = false;
		return hitInfo;
		//return false;
	} else if (pt <st) //plane hit
	{
		hitInfo.t = pt;
		hitInfo.point = ray.getPoint(pt);
		hitInfo.normal = p[maxPi].normal;
		//t = pt;
		hitInfo.phongInfo = p[maxPi].phong;
	}else if (st < pt) //sphere hit
	{
		hitInfo.t = st;
		hitInfo.point = ray.getPoint(st);
		hitInfo.normal = s[maxSi].getNormal(hitInfo.point);
		//t = st;
		hitInfo.phongInfo = s[maxSi].phong;
	}
	hitInfo.hit = true;
	return hitInfo;
	//return true;
}


__device__ Color TraceRay(const Ray &ray,  Plane* p,   Sphere* s,PointLight &light, SceneStats* sceneStats, int recursion)
{

	HitInfo hitInfo=Intersect(ray,p,s,sceneStats);
	if (hitInfo.hit){
		Color color = hitInfo.phongInfo.ambient;
		//light
		Ray sray = light.getShadowRay(hitInfo.point);
		sray.ShiftStart();
		HitInfo sHit = Intersect(sray,p,s,sceneStats);
		if ((sHit.hit) && (sHit.t>1)) {
			float3 lv = sray.direction;
			lv = CUDA::normalize(lv);
			color.accumulate(CUDA::mult(hitInfo.phongInfo.diffuse,light.color),fabs(CUDA::dot(lv,hitInfo.normal)));
			if (hitInfo.phongInfo.shininess != 0) {
				float3 rlv = float3_sub(CUDA::cross(float3_mult(2.f,CUDA::cross(lv,hitInfo.normal)), hitInfo.normal) , lv);
				float3 vv = ray.direction;
				vv = CUDA::normalize(vv);
				float specular = -1.f * CUDA::dot(rlv,vv);
				if (specular > 0) {
					color.accumulate(mult(hitInfo.phongInfo.specular, light.color), pow(specular, hitInfo.phongInfo.shininess));
				}
			}

		}
		//reflected ray
		if ((hitInfo.phongInfo.reflectance>0) && (recursion > 0)) {
			Ray rray(hitInfo.point, float3_sub(ray.direction, CUDA::cross(float3_mult(2,CUDA::cross(ray.direction,hitInfo.normal)) ,hitInfo.normal)));
			rray.ShiftStart(1e-5);

			//Color rcolor = TraceRay(rray, p,s, light,sceneStats, recursion-1);
			//        color *= 1-phong.GetReflectance();
			//color.accumulate(rcolor, hitInfo.phongInfo.reflectance);
		}
		return color;
	}else
	{
		return Color(0.f,0.f,0.f);
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
__global__ void RTKernel(uchar4* data, uint32 width, uint32 height, Sphere* spheres, Plane* planes, SceneStats* sceneStats, Camera* camera)
{
	uint32 X = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32 Y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float x = (2.f*X/WINDOW_WIDTH - 1.f);
	float y = (2.f*Y/WINDOW_HEIGHT - 1.f);

	//float dx = 2.0/width;
	//float dy = 2.0/height;

	Ray ray = camera->getRay(x,y);

	PointLight l(make_float3(4.f, 5.f, 3.f),Color(1.0,1.0,1.0));
	//Color c = TraceRay(ray,scene,l,15);
	Color c = TraceRay(ray, planes, spheres, l,sceneStats, 5);


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

	data[WINDOW_WIDTH * Y + X].x = min(c.red*255.f,255.f);
	data[WINDOW_WIDTH * Y + X].y = min(c.green*255.f,255.f);
	data[WINDOW_WIDTH * Y + X].z = min(c.blue*255.f,255.f);
	data[WINDOW_WIDTH * Y + X].w = 0;
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
extern "C" void launchRTKernel(uchar4* data, uint32 imageWidth, uint32 imageHeight, Sphere* spheres, Plane* planes, SceneStats* sceneStats, Camera* camera)
{   	
	dim3 threadsPerBlock(8, 8, 1); // 64 threads ~ 8*8
	dim3 numBlocks(WINDOW_WIDTH / threadsPerBlock.x, WINDOW_HEIGHT / threadsPerBlock.y);

	RTKernel<<<numBlocks, threadsPerBlock>>>(data, imageWidth, imageHeight, spheres, planes, sceneStats, camera);

	cudaThreadSynchronize();
	checkCUDAError();
}
