#include "Ray.h"

Ray::Ray()
{

}

Ray::Ray(const Vector3& origin, const Vector3& direction)
{
	this->origin = origin;
	this->direction = direction;
}

Vector3 Ray::At(double t) const
{
	return origin + (direction*t);
}

Vector3 Ray::GetOrigin() const
{
	return origin;
}

Vector3 Ray::GetDirection() const
{
	return direction;
}
