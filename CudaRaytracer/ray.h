#ifndef RAY_H
#define RAY_H

#include "mathematics.h"

struct Ray {
	__device__ Ray()
	{}

	__device__ Ray(float3 const& o, float3 d)
		: origin(o), direction(CUDA::normalize(d))
	{}
    __device__ float3 getPoint(double t) const { return CUDA::float3_add(origin, CUDA::float3_mult(t, direction)); }
	float3 origin, direction;
};

#endif
