#ifndef MATHEMATICS_H
#define MATHEMATICS_H

#include <vector_types.h>
#include <cmath>

namespace math {

__device__ inline float devScalarProduct(float3 const& v1, float3 const& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

	
__device__ inline float3 devCrossProduct(float3 const& v1, float3 const& v2)
{	
	return make_float3(v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x);	
}

inline float scalarProduct(float3 const& v1, float3 const& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

	
inline float3 crossProduct(float3 const& v1, float3 const& v2)
{	
	return make_float3(v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x);	
}

template <typename T>
__device__ inline bool devInsideInterval(T num, T l, T r, bool included = false)
{
	return included ? (num >= l && num <= r) : (num > l && num < r);	
}

__device__ inline void devNormalize(float3& vector)
{
	float magnitude = sqrt((vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z));

	vector.x /= magnitude;
	vector.y /= magnitude;
	vector.z /= magnitude;	
}

inline void normalize(float3& vector)
{
	float magnitude = sqrt((vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z));

	vector.x /= magnitude;
	vector.y /= magnitude;
	vector.z /= magnitude;	
}

template <typename T>
__device__ inline T min(T const& a, T const& b)
{
	return a > b ? b : a;
}

}


__device__ inline float3 devmul(float3 const& vector, float const& f)
{
	return make_float3(vector.x * f, vector.y * f, vector.z * f);
}

__device__ inline float3 devmul(float const& f, float3 const& vector)
{
	return make_float3(vector.x * f, vector.y * f, vector.z * f);
}

__device__ inline float3 devsub(float3 const& l, float3 const& r)
{ 
	return make_float3(l.x - r.x, l.y - r.y, l.z - r.z);	
}

__device__ inline float3 devadd(float3 const& l, float3 const& r)
{ 
	return make_float3(l.x + r.x, l.y + r.y, l.z + r.z);	
}

inline float3 operator* (float3 const& vector, float const& f)
{
	return make_float3(vector.x * f, vector.y * f, vector.z * f);
}

inline float3 operator* (float const& f, float3 const& vector)
{
	return vector * f;
}

inline float3 operator- (float3 const& l, float3 const& r)
{ 
	return make_float3(l.x - r.x, l.y - r.y, l.z - r.z);	
}

inline float3 operator+ (float3 const& l, float3 const& r)
{ 
	return make_float3(l.x + r.x, l.y + r.y, l.z + r.z);	
}

#endif
