<<<<<<< HEAD
#include <vector_types.h>

namespace CUDA {

		struct Plane
		{
			__device__ Plane(float3 n, float3 point)				
			{
				normal = n;
				math::normalize(normal);
				d = 1.f * math::dot(normal, point);
			}

			__device__ bool intersect(Ray const& ray) {
				float np = math::dot(normal, ray.direction);

				if (np != 0) {
					float t = -1.f * (d * math::dot(normal, ray.origin)) / np;
					if (t > 0) {
						return true;
					}
				}
				return false;
			}

			float3 normal;
			float d;
		};

}
=======
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
>>>>>>> origin/master
