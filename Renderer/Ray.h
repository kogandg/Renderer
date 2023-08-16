#pragma once

#include "Vector3.h"

class Ray
{
private:
	Vector3 origin;
	Vector3 direction;
public:
	Ray();
	Ray(const Vector3& origin, const Vector3& direction);

	Vector3 At(double t) const;

	Vector3 GetOrigin() const;
	Vector3 GetDirection() const;
};

