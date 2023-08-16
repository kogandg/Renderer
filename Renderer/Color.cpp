#include "Color.h"

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
}

Color::Color(double r, double g, double b, double a)
{
	SetR(r);
	SetG(g);
	SetB(b);
	SetA(a);
	bytesPerColor = 4;
}

Color::Color(double r, double g, double b)
{
	SetR(r);
	SetG(g);
	SetB(b);
	a = 255;
	bytesPerColor = 4;
}

Color::Color(int val, int bpc)
{
	value = val;
	bytesPerColor = bpc;
}

Color::Color(const Color& c)
{
	value = c.value;
	bytesPerColor = c.bytesPerColor;
}

Color::Color(const unsigned char* p, int bpc)
{
	value = 0;
	bytesPerColor = bpc;
	for (int i = 0; i < bpc; i++)
	{
		raw[i] = p[i];
	}
}

unsigned char* Color::GetRaw()
{
	return raw;
}

void Color::SetRaw(unsigned char raw[])
{
	for (int i = 0; i < bytesPerColor; i++)
	{
		this->raw[i] = raw[i];
	}
}

double Color::GetR() const
{
	return (double)(r)/255.0f;
}

double Color::GetG() const
{
	return (double)(g) / 255.0f;
}

double Color::GetB() const
{
	return (double)(b) / 255.0f;
}

double Color::GetA() const
{
	return (double)(a) / 255.0f;
}

void Color::SetR(double r)
{
	this->r = r * 255;
}

void Color::SetG(double g)
{
	this->g = g * 255;
}

void Color::SetB(double b)
{
	this->b = b * 255;
}

void Color::SetA(double a)
{
	this->a = a * 255;
}

Color& Color::operator=(const Color& c)
{
	if (this != &c)
	{
		bytesPerColor = c.bytesPerColor;
		value = c.value;
	}

	return *this;
}

Color Color::operator*(double t)
{
	return Color(GetR()*t, GetG()*t, GetB()*t);
}

Color Color::operator+(const Color& obj)
{
	return Color(GetR() + obj.GetR(), GetG() + obj.GetG(), GetB() + obj.GetB());
}

