#ifndef PHONG_H
#define PHONG_H

#include "color.h"

struct PhongInfo
{
  Color diffuse;
  Color specular;
  Color ambient;
  float shininess;
  float reflectance;
  PhongInfo(const Color &diff, const Color &spec, const Color &amb, float shin, float ref = 0.0) {
    diffuse = diff; specular = spec; ambient = amb; shininess = shin;
    reflectance = ref;
  }
  __device__ PhongInfo()
  {};
};

struct HitInfo {
	bool hit;
	float3 point;
	float3 normal;
	PhongInfo phongInfo;
};



#endif