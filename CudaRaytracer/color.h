#ifndef COLOR_H
#define COLOR_H

struct rgb
{
	public:		
		rgb() 
			: _r(0.0f), _g(0.0f), _b(0.0f)
		{}
		rgb(float const& red, float const& green, float const& blue) 
			: _r(red), _g(green), _b(blue) 
		{}
		rgb( rgb const& color )
		{ _r = color.red(); _g = color.green(); _b = color.blue(); }

		void setRed(float value){ _r = value; }
		void setGreen(float value){ _g = value; }
		void setBlue(float value){ _b = value; }

		float red() const { return _r; }
		float green() const { return _g; }
		float blue() const { return _b; }

		rgb operator* ( rgb const& color )
		{ return rgb(_r * color.red(), _g * color.green(), _b * color.blue()); }

		rgb operator+ ( rgb const& color )
		{ return rgb(_r + color.red(), _g + color.green(), _b + color.blue()); }

		rgb& operator+= ( rgb const& color )
		{ _r += color.red(); _g += color.green(); _b += color.blue(); return (*this); }

	private:
		float _r, _g, _b;
};

inline rgb operator* ( float const& coef, rgb const& color )
{ return rgb( coef * color.red(), coef * color.green(), coef * color.blue() ); }

inline rgb operator* ( rgb const& color, float const& coef )
{ return coef * color; }

inline rgb operator/ ( rgb const& color, float const& coef )
{ return rgb( color.red() / coef, color.green() / coef, color.blue() / coef ); }


#endif
