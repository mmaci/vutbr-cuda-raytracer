#ifndef RAY_H
#define RAY_H

#include "vector3.h"

struct ray {
	public:
		ray() 
		{}

		ray(vector3 origin, vector3 direction) :
			_origin(origin), _direction(direction)
		{}

		public void setOrigin(vector3 origin) { _origin = origin; }

		public void setDirection(vector3 direction) { _direction = direction; }

	private:
		vector3 _origin;
		vector3 _direction;
};

#endif
