#ifndef RAY_H
#define RAY_H

#include <vector_types.h>
#include "mathematics.h"

namespace CUDA {

		struct Ray {
			__device__ Ray()
			{}

			__device__ Ray(float3 const& o, float3 d)
				: origin(o), direction(math::normalize(d))
			{}
            __device__ float3 getPoint(double t) const { return origin + t*direction; }
			float3 origin, direction;
		};

}

#endif
