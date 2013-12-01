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

	__device__ HitInfo intersect(Ray const& ray) {
		float np = CUDA::dot(normal, ray.direction);
		HitInfo hit;
		if (np != 0) {
			float t = -1.f * (d * CUDA::dot(normal, ray.origin)) / np;
			if (t > 0) {
				hit.hit = true;
				hit.t = t;
				return hit;
			}
		}
		hit.hit = false;
		return hit;
	}			

	float3 normal;
	float d;
	PhongInfo phong;
};



#endif
