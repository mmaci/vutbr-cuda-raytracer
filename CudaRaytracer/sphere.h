#ifndef SPHERE_H
#define SPHERE_H

#include <vector_types.h>

namespace CUDA {

		struct Sphere
		{
			__device__ Sphere()
			{}
			__device__ Sphere(float3 const& c, float const& r) :
				center(c), radius(r)
			{}
		
			__device__ bool intersect(Ray const& ray) {
				float aa = math::dot(ray.direction, ray.direction);
				float bb = 2.f * (math::dot(ray.direction, (ray.origin - center)));
				float cc = math::dot((ray.origin - center), (ray.origin - center)) - (radius * radius);

				float D = bb * bb - 4 * aa * cc;

				if (D > 0) {
					float sD = sqrt(D);
					float t0 = (-bb - sD) / (2 * aa);
					if (t0 > 0.f)
						return true;
					float t1 = (-bb + sD) / (2 * aa);
					if (t1 > 0.f)
						return true;
				}
				return false;
			}

			float3 center;
			float radius;	
		};
}

#endif
