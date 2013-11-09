#ifndef SPHERE_H
#define SPHERE_H

#include <vector_types.h>

namespace CUDA {
	namespace geometry {

		struct Sphere
		{
			__device__ Sphere()
			{}
			__device__ Sphere(float3 const& o, float const& r) :
				origin(o), radius(r)
			{}
			__device__ Sphere(float const& x, float const& y, float const& z, float const& r) :
				origin(make_float3(x, y, z)), radius(r)
			{}

			float3 origin;
			float radius;	
		};
	}
}

#endif
