#include "Sphere.h"

Sphere::Sphere(Vector3 center, double radius, shared_ptr<Material> material)
{
    this->center = center;
    this->radius = radius;
	this->material = material;
}

bool Sphere::Hit(const Ray& ray, Interval rayT, HitRecord& record) const
{
	Vector3 oc = ray.GetOrigin() - center;
	//quaradic equation
	double a = ray.GetDirection().LengthSquared();
	double halfB = oc.Dot(ray.GetDirection());
	double c = oc.LengthSquared() - (radius * radius);

	double discriminant = (halfB * halfB) - (a * c);
	if (discriminant < 0)
	{
		return false;
	}
	double sqrtd = sqrt(discriminant);
	
	//find nearest root that is in acceptable range
	double root = (-halfB - sqrtd) / a;
	if (!rayT.Surrounds(root))
	{
		root = (-halfB + sqrtd) / a;
		if (!rayT.Surrounds(root))
		{
			return false;
		}
	}

	record.T = root;
	record.Position = ray.At(record.T);
	Vector3 outwardNormal = (record.Position - center) / radius;
	record.SetFaceNormal(ray, outwardNormal);
	record.Material = material;

	return true;
}
