#include "Color.h"

void Color::matchRaws()
{
	r = R * 255;
	g = G * 255;
	b = B * 255;
	a = A * 255;
}

void Color::matchReals()
{
	R = (double)(r) / 255.0;
	G = (double)(g) / 255.0;
	B = (double)(b) / 255.0;
	A = (double)(a) / 255.0;
}


Color::Color()
{
	value = 0;
	bytesPerColor = 1;
}


Color::Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	this->r = r;
	this->g = g;
	this->b = b;
	this->a = a;
	bytesPerColor = 4;
	matchReals();
}

Color::Color(double r, double g, double b, double a)
{
	R = r;
	G = g;
	B = b;
	A = a;
	bytesPerColor = 4;
	matchRaws();
}

Color::Color(double r, double g, double b)
{
	R = r;
	G = g;
	B = b;
	A = 1;
	bytesPerColor = 4;
	matchRaws();
}

Color::Color(int val, int bpc)
{
	value = val;
	bytesPerColor = bpc;
	matchReals();
}

Color::Color(const Color& c)
{
	R = c.R;
	G = c.G;
	B = c.B;
	A = c.A;
	bytesPerColor = c.bytesPerColor;
	matchRaws();
}

Color::Color(const unsigned char* p, int bpc)
{
	value = 0;
	bytesPerColor = bpc;
	for (int i = 0; i < bpc; i++)
	{
		raw[i] = p[i];
	}
	matchReals();
}

unsigned char* Color::GetRaw()
{
	matchRaws();
	return raw;
}

void Color::SetRaw(unsigned char raw[])
{
	for (int i = 0; i < 4; i++)
	{
		this->raw[i] = raw[i];
	}
	matchReals();
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
		A = c.A;
		bytesPerColor = c.bytesPerColor;
		matchRaws();
	}

	return *this;
}
Color Color::operator*(const Color& obj) const
{
	return Color(R * obj.R, G * obj.G, B * obj.B);
}

Color Color::operator*(double t)
{
	return Color(R*t, G*t, B*t);
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
