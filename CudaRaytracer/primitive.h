#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "material.h"
#include "vector3.h"

class Primitive
{
	public:
		Primitive()
		{}	

		void setMaterial(material const& m)
		{ _material = m; }

		material getMaterial() const
		{ return _material; }

	private:
		material _material;		
};

#endif
