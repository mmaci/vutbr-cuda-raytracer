#ifndef SPHERE_H
#define SPHERE_H

#include <vector_types.h>
#include "color.h"
#include "mathematics.h"

namespace CUDA {

		struct Sphere
		{
			__device__ Sphere()
			{}
			__device__ Sphere(float3 const& ce, float const& r,Color c) :
				color(c), center(ce), radius(r)
			{}
		
			__device__ float intersect(Ray const& ray) {
				float aa = math::dot(ray.direction, ray.direction);
				float bb = 2.f * (math::dot(ray.direction, (ray.origin - center)));
				float cc = math::dot((ray.origin - center), (ray.origin - center)) - (radius * radius);

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
				return 0.0;
			}

			__device__ float3 GetNormal(const float3 &position) {
				float3 n = position - center;
				math::normalize(n);
				return n;
			}
			float3 center;
			float radius;
			Color color;
		};
}

#endif
