#ifndef LIGHT_H
#define LIGHT_H

#include "ray.h"
#include "color.h"
#include <vector_types.h>


namespace CUDA
{
	struct PointLight
	{
	  float3 p;
      Color c;
	  __device__ PointLight(const float3 &position, const Color &color) { p = position; c = color; }
	  __device__ Ray GetShadowRay(const float3 &point) { return Ray(point, p - point); }
      __device__ Color GetColor() { return c; }
	
	
	};




}
#endif