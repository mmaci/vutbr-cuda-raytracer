#ifndef COLOR_H
#define COLOR_H

#include "mathematics.h"

struct Color
{			
	__device__ Color()
		: red(0.f), green(0.f), blue(0.f)
	{ }
	__device__ Color(float const& ared, float const& agreen, float const& ablue)
		: red(ared), green(agreen), blue(ablue)
	{ }
	__device__ void accumulate(Color const& x, double const& scale = 1.0) 
	{ red += scale*x.red; green += scale*x.green; blue += scale*x.blue; }

	__device__ Color &operator *= (float const& x) { red *= x; green *= x; blue *= x; return *this; }

	float red;
	float green;
	float blue;
};


namespace CUDA{	
	__device__ inline Color mult(const Color &a, const Color &b){ return Color(a.red*b.red, a.green*b.green, a.blue*b.blue); }	
	__device__ inline float diff(const Color &a, const Color &b){
		float rr = a.red - b.red;
		float gg = a.green - b.green;
		float bb = a.blue - b.blue;
		return rr*rr + gg*gg + bb*bb;
	}

}
#endif
