#ifndef LIGHT_H
#define LIGHT_H

#include "ray.h"
#include "color.h"
#include <vector_types.h>


namespace CUDA
{
	struct PointLight
	{	  
		__device__ PointLight(float3 const& p, Color const& c)
			: position(p), color(c)
		{ }
		__device__ Ray getShadowRay(float3 const& point) { return Ray(point, position - point); }		
	
		float3 position;
		Color color;
	};




}
#endif