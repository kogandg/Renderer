#pragma once
#include "Helpers.h"

class HitRecord
{
public:
	Vector3 Position;
	Vector3 Normal;
	double T;
	bool FrontFace;

	void SetFaceNormal(const Ray& ray, const Vector3& outwardNormal);
};

class Hittable
{
public:
	//virtual ~Hittable() = default;
	virtual bool Hit(const Ray& ray, Interval rayT, HitRecord& rec) const = 0;
};

