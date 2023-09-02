#include "Color.cuh"

Color::Color()
{
	R = 0;
	G = 0;
	B = 0;
}

Color::Color(double r, double g, double b)
{
	R = r;
	G = g;
	B = b;
}

Color::Color(const Color& c)
{
	R = c.R;
	G = c.G;
	B = c.B;
}

Color Color::LinearToGamma()
{
	return Color(sqrt(R), sqrt(G), sqrt(B));
}

Color& Color::operator=(const Color& c)
{
	if (this != &c)
	{
		R = c.R;
		G = c.G;
		B = c.B;
	}

	return *this;
}
Color Color::operator*(const Color& obj) const
{
	return Color(R * obj.R, G * obj.G, B * obj.B);
}

Color Color::operator*(double t)
{
	return Color(R * t, G * t, B * t);
}

Color Color::operator+(const Color& obj)
{
	return Color(R + obj.R, G + obj.G, B + obj.B);
}

Color& Color::operator+=(const Color& rhs)
{
	*this = *this + rhs;
	return *this;
}