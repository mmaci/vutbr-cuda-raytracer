#ifndef SPHERE_H
#define SPHERE_H

#include "phong.h"

#include "mathematics.h"

#include "ray.h"

struct Sphere
{
	Sphere()
	{}
	Sphere(float3 const& ce, float const& r, PhongInfo p) :
		phong(p), center(ce), radius(r)
	{}
		
	__device__ float intersect(Ray const& ray) {
		float aa = CUDA::dot(ray.direction, ray.direction);
		float bb = 2.f * (CUDA::dot(ray.direction, CUDA::float3_sub(ray.origin, center)));
		float cc = CUDA::dot(CUDA::float3_sub(ray.origin, center), CUDA::float3_sub(ray.origin, center)) - (radius * radius);

		float D = bb * bb - 4 * aa * cc;

		if (D > 0) {
			float sD = sqrt(D);
			float t0 = (-bb - sD) / (2 * aa);
			if (t0 > 0.f)
				return t0;
			float t1 = (-bb + sD) / (2 * aa);
			if (t1 > 0.f)
				return t1;
		}
		return 0.f;
	}

	__device__ float3 getNormal(float3 const& position) const {
		float3 n = CUDA::float3_sub(position, center);
		CUDA::normalize(n);
		return n;
	}

	float3 center;
	float radius;
	PhongInfo phong;
};


#endif
