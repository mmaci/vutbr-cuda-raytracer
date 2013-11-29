#ifndef MATHEMATICS_H
#define MATHEMATICS_H

#include <vector_types.h>
#include <cmath>

namespace CUDA {
	namespace math {

		__device__ inline float dot(float3 const& v1, float3 const& v2)
		{
			return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
		}

	
		__device__ inline float3 cross(float3 const& v1, float3 const& v2)
		{	
			return make_float3(v1.y * v2.z - v1.z * v2.y,
				v1.z * v2.x - v1.x * v2.z,
				v1.x * v2.y - v1.y * v2.x);	
		}

		__device__ inline float3& normalize(float3& vector)
		{
			float len = sqrt((vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z));

			vector.x /= len;
			vector.y /= len;
			vector.z /= len;	

			return vector;
		}
	}
}



__device__ inline float3 operator- (float3 const& l, float3 const& r)
{ 
	return make_float3(l.x - r.x, l.y - r.y, l.z - r.z);	
}

__device__ inline float3 operator+ (float3 const& l, float3 const& r)
{ 
	return make_float3(l.x + r.x, l.y + r.y, l.z + r.z);	
}

__device__ inline float3 operator* (float const& l, float3 const& r)
{ 
	return make_float3(l * r.x, l * r.y, l * r.z);	
}

#endif
