#ifndef SPHERE_H
#define SPHERE_H

#include <vector_types.h>

struct Sphere
{
	__device__ Sphere()
	{}

	__device__ Sphere(float3 const& c, float const& r)
		: center(c), radius(r)
	{}

	float3 center;
	float radius;
};

#endif
