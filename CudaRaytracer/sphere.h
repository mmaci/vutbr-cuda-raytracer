#ifndef SPHERE_H
#define SPHERE_H

#include "primitive.h"

class Sphere : public Primitive
{
	public:
		Sphere(vector3 const& center, float const& r)
			: _center(center), _radius(r)
		{ }		

	private:
		vector3 _center;
		float _radius;
};

#endif
