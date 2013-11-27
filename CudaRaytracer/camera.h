#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

namespace CUDA {

struct Camera {

	__device__ Camera() : 
		position(make_float3(0.f, 0.f, 0.f)), direction(make_float3(0.f, 0.f, 1.f)), right(make_float3(1.f, 0.f, 0.f)), up(make_float3(0.f, 1.f, 0.f))
	{}

	__device__ void lookAt(float3 const& eye, float3 const& target, float3 const& sky, float fov, float ratio) {
		position = eye;
		direction = math::normalize(target-eye);
		right = ratio * math::normalize(math::cross(sky, direction));
		up = math::normalize(math::cross(direction, right));
	}

	__device__ Ray getRay(float u, float v) const {
		return Ray(position, direction + u*right + v*up);
	}

	float3 position;
	float3 direction;
	float3 right;
	float3 up;

};

}

#endif
