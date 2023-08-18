#pragma once
#include <math.h>
#include "Helpers.h"

using namespace std;

class Vector3
{
public:
	double X;
	double Y;
	double Z;

	Vector3();
	Vector3(double x, double y, double z);
	Vector3(const Vector3& v);

	double Length();
	double LengthSquared();

	double Dot(const Vector3& v);
	Vector3 Cross(const Vector3& v);

	Vector3 Unit();

	Vector3 operator-() const;

	Vector3& operator+=(const Vector3& rhs);

	Vector3& operator*=(double t);
	Vector3& operator/=(double t);

	Vector3 operator+(const Vector3& obj) const;
	Vector3 operator-(const Vector3& obj) const;
	Vector3 operator*(const Vector3& obj) const;

	Vector3 operator*(double t) const;
	Vector3 operator/(double t) const;

	static Vector3 Random();
	static Vector3 Random(double min, double max);
	static Vector3 RandomInUnitSphere();
	static Vector3 RandomUnitVector();
	static Vector3 RandomOnHemisphere(const Vector3& normal);
};

