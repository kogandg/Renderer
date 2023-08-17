#pragma once
#include "Hittable.h"
#include "Helpers.h"

class Sphere : public Hittable
{
private:
	Vector3 center;
	double radius;

public:
	Sphere(Vector3 center, double radius);

	bool Hit(const Ray& ray, Interval rayT, HitRecord& rec) const override;
};

