#pragma once
#include "Hittable.h"
#include "Helpers.h"
#include "Vector3.h"
#include "Interval.h"
#include "Ray.h"


class Sphere : public Hittable
{
private:
	Vector3 center;
	double radius;
	shared_ptr<Material> material;

public:
	Sphere(Vector3 center, double radius, shared_ptr<Material> material);

	bool Hit(const Ray& ray, Interval rayT, HitRecord& rec) const override;
};

