#ifndef PLANE_H
#define PLANE_H

#include <vector_types.h>
#include "mathematics.h"

namespace CUDA {
	namespace geometry {

		struct Plane
		{
			__device__ Plane(float3 const& normal, float3 const& point)
			{
				n = normal;
				math::normalize(n);
				d = -(math::dot(n,point));	
			}
			float3 n;
			float d;	
		};
	}
}

#endif
