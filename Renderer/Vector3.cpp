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

bool Vector3::NearZero() const
{
	double small = 1e-8;
	return (fabs(X) < small) && (fabs(Y) < small) && (fabs(Z) < small);
}

double Vector3::Dot(const Vector3& v)
{
	return (X * v.X) + (Y * v.Y) + (Z * v.Z);
}

Vector3 Vector3::Cross(const Vector3& v)
{
	return Vector3((Y * v.Z) - (Z * v.Y), (Z * v.X) - (X * v.Z), (X * v.Y) - (Y * v.X));
}

Vector3 Vector3::Reflect(const Vector3& normal)
{
	return *this - normal*Dot(normal)*2;
}

Vector3 Vector3::Refract(const Vector3& normal, double etaiOverEtat)
{
	double cosTheta = fmin((-(*this)).Dot(normal), 1);
	Vector3 rOutPerpendicular = (*this + normal * cosTheta) * etaiOverEtat;
	Vector3 rOutParallel = normal * -sqrt(fabs(1.0-rOutPerpendicular.LengthSquared()));
	return rOutPerpendicular + rOutParallel;
}

Vector3 Vector3::Unit()
{
	double length = Length();
	if (length == 0)
	{
		return Vector3(0, 0, 0);
	}
	return *this / length;
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

Vector3 Vector3::Random()
{
	return Vector3(randomDouble(), randomDouble(), randomDouble());
}

Vector3 Vector3::Random(double min, double max)
{
	return Vector3(randomDouble(min, max), randomDouble(min, max), randomDouble(min, max));
}

Vector3 Vector3::RandomInUnitDisk()
{
	while (true)
	{
		Vector3 point = Vector3(randomDouble(-1, 1), randomDouble(-1, 1), 0);
		if (point.LengthSquared() < 1)
		{
			return point;
		}
	}
}

Vector3 Vector3::RandomInUnitSphere()
{
	while (true)
	{
		Vector3 point = Random(-1, 1);
		if (point.LengthSquared() < 1)
		{
			return point;
		}
	}
}

Vector3 Vector3::RandomUnitVector()
{
	return RandomInUnitSphere().Unit();
}

Vector3 Vector3::RandomOnHemisphere(const Vector3& normal)
{
	Vector3 onUnitSphere = RandomUnitVector();
	if (onUnitSphere.Dot(normal) > 0.0)
	{
		return onUnitSphere;
	}
	return -onUnitSphere;
}







