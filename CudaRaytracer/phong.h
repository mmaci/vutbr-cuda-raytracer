#ifndef PHONG_H
#define PHONG_H

#include "color.h"

struct HitInfo {
	bool hit;
	float3 point;
	float3 normal;
	Color color;
};


#endif