#include "Vector3.h"

Vector3::Vector3()
{
	X = 0;
	Y = 0;
	Z = 0;
}

Vector3::Vector3(double x, double y, double z)
{
	X = x;
	Y = y;
	Z = z;
}

Vector3::Vector3(const Vector3& v)
{
	X = v.X;
	Y = v.Y;
	Z = v.Z;
}

double Vector3::Length()
{
	return sqrt(LengthSquared());
}

double Vector3::LengthSquared()
{
	return (X * X) + (Y * Y) + (Z * Z);
}

double Vector3::Dot(const Vector3& v)
{
	return (X * v.X) + (Y * v.Y) + (Z * v.Z);
}

Vector3 Vector3::Cross(const Vector3& v)
{
	return Vector3((Y * v.Z) - (Z * v.Y), (Z * v.X) - (X * v.Z), (X * v.Y) - (Y * v.X));
}

Vector3 Vector3::Unit()
{
	double length = Length();
	return Vector3(X/length, Y/length, Z/length);
}

Vector3 Vector3::operator-() const
{
	return Vector3(-X, -Y, -Z);
}

Vector3& Vector3::operator+=(const Vector3& rhs)
{
	X += rhs.X;
	Y += rhs.Y;
	Z += rhs.Z;
	return *this;
}

Vector3& Vector3::operator*=(double t)
{
	X *= t;
	Y *= t;
	Z *= t;
	return *this;
}

Vector3& Vector3::operator/=(double t)
{
	return *this *= 1 / t;
}

Vector3 Vector3::operator+(const Vector3& obj) const
{
	return Vector3(X + obj.X, Y + obj.Y, Z + obj.Z);
}

Vector3 Vector3::operator-(const Vector3& obj) const
{
	return Vector3(X - obj.X, Y - obj.Y, Z - obj.Z);
}

Vector3 Vector3::operator*(const Vector3& obj) const
{
	return Vector3(X * obj.X, Y * obj.Y, Z * obj.Z);
}

Vector3 Vector3::operator*(double t) const
{
	return Vector3(X * t, Y * t, Z * t);
}

Vector3 Vector3::operator/(double t) const
{
	return Vector3(X / t, Y / t, Z / t);
}







