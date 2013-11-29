#ifndef COLOR_H
#define COLOR_H

namespace CUDA {

		struct Color
		{	
			float red;
			float green;
			float blue;

			__device__ Color() { red = 0; green = 0; blue = 0; }
			__device__ Color(float ared, float agreen, float ablue) { red = ared; green = agreen; blue = ablue; }
			__device__ inline void Accumulate(const Color &x, double scale = 1.0) {
				red += scale*x.red; green += scale*x.green; blue += scale *x.blue;

			}
			__device__ Color &operator *= (float x) { red *= x; green *= x; blue *= x; return Color(red,green,blue); }
		};

}

#endif
