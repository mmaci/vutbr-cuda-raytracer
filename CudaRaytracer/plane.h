#ifndef PLANE_H
#define PLANE_H

#include "phong.h"

struct Plane
{
	Plane(float3 n, float3 point, PhongInfo p)	
		: phong(p), normal(n)
	{				
		normalize(normal);
		d = 1.f * dot(normal, point);
	}

	__device__ float intersect(Ray const& ray) {
		float np = CUDA::dot(normal, ray.direction);

		if (np != 0) {
			float t = -1.f * (d * CUDA::dot(normal, ray.origin)) / np;
			if (t > 0) {
				return t;
			}
		}
		return 0;
	}			

	float3 normal;
	float d;
	PhongInfo phong;
};



#endif
