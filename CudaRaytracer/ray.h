#ifndef RAY_H
#define RAY_H

#include <vector_types.h>
#include "mathematics.h"

struct Ray {

	__device__ Ray()
	{}

	__device__ Ray(float3 const& o, float3 const& d)
		: origin(o), dir(d)
	{
		math::devNormalize(dir);
	}

	float3 origin, dir;

};

#endif
