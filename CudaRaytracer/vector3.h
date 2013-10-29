#ifndef MATHEMATICS_H
#define MATHEMATICS_H

#include <cmath>

struct vector3
{
	public:
		vector3()
		{}
		vector3(float const& x, float const& y, float const& z)
			: _x(x), _y(y), _z(z)
		{}

		float x() const { return _x; }
		float y() const { return _y; }
		float z() const { return _z; }

		void setX( float val ){ _x = val; }
		void setY( float val ){ _y = val; }
		void setZ( float val ){ _z = val; }
		
		vector3 operator- ( vector3 const& v ) const
		{ return vector3( _x - v.x(), _y - v.y(), _z - v.z() ); }

		vector3 operator+ ( vector3 const& v ) const
		{ return vector3( _x + v.x(), _y + v.y(), _z + v.z() ); }

		float length() const
		{ return sqrt( _x*_x + _y*_y + _z*_z ); }

		vector3& normalize()
		{
			float size = sqrt( _x*_x + _y*_y + _z*_z );

			if (size == 0 || size == 1)
				return *this;

			_x /= size;
			_y /= size;
			_z /= size;

			return *this;
		}			

		bool isNull() const
		{
			return ( _x <= 0.0f && _y <= 0.0f && _z <= 0.0f );
		}

	private:
		float _x, _y, _z;
};


inline vector3 operator* ( vector3 const& v, float const& n )
{ return vector3(n * v.x(), n * v.y(), n * v.z()); }

inline vector3 operator* ( float const& n, vector3 const& v )
{ return v * n; }

inline vector3 operator* ( vector3 const& u, vector3 const& v )
{ return vector3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z()); }

inline vector3 operator/ ( vector3 const& v, float const& n )
{ return vector3(v.x() / n, v.y() / n, v.z() / n); }

#endif
