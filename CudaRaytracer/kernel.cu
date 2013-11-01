#include <vector_types.h>
#include <vector_functions.h>
#include <math_functions.h>

#include "constants.h"
#include "ray.h"
#include "sphere.h"



__device__ bool intersectSphere(Ray const& ray, Sphere const& sphere, float &t)
{
	float b, c, d;
	
	float3 sr = devsub(ray.origin, sphere.center);

	b = math::devScalarProduct(sr, ray.dir);
	c = math::devScalarProduct(sr, sr) - (sphere.radius * sphere.radius);
	d = b*b - c;
	if (d > 0) 
	{
		float e = sqrt(d);
		float t0 = -b-e;
		if(t0 < 0)
			t = -b+e;
		else
			t = math::min(-b-e,-b+e);

		return true;
	}
	return false;
}


__global__ void rayTraceKernel(float3* buffer, Sphere* spheres, uint32 numSpheres,
					const float3 a, const float3 b, const float3 c,
					const float3 campos)
{	
	uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;

	float xf = (x-0.5f) / (static_cast<float>(WIDTH));
	float yf = (y-0.5f) / (static_cast<float>(HEIGHT));	
	

	float3 t1 = devadd(c, (devmul(a, xf)));
	float3 t2 = devmul(b, yf);

	Ray ray(devadd(t1, t2), devsub(devadd(t1, t2), campos));				

	// check all spheres if they intersect with the generated ray
	float hit;
	for (uint8 i = 0; i < numSpheres; ++i)
	{
		if ((x*WIDTH + y) > 1000)
		{
			buffer[x*WIDTH + y].x = 1.0f;		
			buffer[x*WIDTH + y].y = 0.0f;		
			buffer[x*WIDTH + y].z = 0.0f;
		}
		if (intersectSphere(ray, spheres[i], hit))
		{
			buffer[x*WIDTH + y].x = 1.0f;		
			buffer[x*WIDTH + y].y = 0.0f;		
			buffer[x*WIDTH + y].z = 0.0f;		
		}
	}	
}

extern "C" 
{
	void RayTraceImage(float3* dataOut, Sphere* spheres, uint32 numSpheres,
		               float3 a, float3 b, float3 c, 
		               float3 campos)
	{
		dim3 block(8,8,1);
		dim3 grid(WIDTH/block.x,HEIGHT/block.y, 1);
		rayTraceKernel<<<grid, block>>>(dataOut, spheres, numSpheres, a ,b, c, campos);
	}

}
