#ifndef PLANE_H
#define PLANE_H

#include <vector_types.h>
#include "mathematics.h"
#include "color.h"

namespace CUDA {

		struct Plane
		{
			__device__ Plane(float3 n, float3 point, Color c)				
			{
				color = c;
				normal = n;
				math::normalize(normal);
				d = 1.f * math::dot(normal, point);
			}

			__device__ float intersect(Ray const& ray) {
				float np = math::dot(normal, ray.direction);

				if (np != 0) {
					float t = -1.f * (d * math::dot(normal, ray.origin)) / np;
					if (t > 0) {
						return t;
					}
				}
				return 0;
			}
			__device__ float3 GetNormal(float3 point = make_float3(0,0,0)){
				return normal;
				
			}

			float3 normal;
			float d;
			Color color;
		};

}

#endif
