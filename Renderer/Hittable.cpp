#include "Hittable.h"

void HitRecord::SetFaceNormal(const Ray& ray, const Vector3& outwardNormal)
{
	//outward normal is assumed to have unit length
	FrontFace = ray.GetDirection().Dot(outwardNormal) < 0;
	Normal = FrontFace ? outwardNormal : -outwardNormal;
}
