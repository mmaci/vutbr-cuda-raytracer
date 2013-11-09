#ifndef RAY_H
#define RAY_H

#include <vector_types.h>
#include "mathematics.h"

namespace CUDA {
	namespace geometry {

		struct Ray {
			__device__ Ray()
			{}

			__device__ Ray(float3 const& o, float3 const& d)
				: origin(o), direction(d)
			{}

			float3 origin, direction;
		};

	}
}

#endif
