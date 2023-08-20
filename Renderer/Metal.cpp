#include "Metal.h"

Metal::Metal(const Color& albedo, double fuzz)
{
    this->albedo = albedo;
    this->fuzz = fuzz;
}

bool Metal::Scatter(const Ray& rayIn, const HitRecord& record, Color& attenuation, Ray& scattered) const
{
    Vector3 reflected = rayIn.GetDirection().Unit().Reflect(record.Normal);
    scattered = Ray(record.Position, reflected + Vector3::RandomUnitVector()*fuzz);
    attenuation = albedo;
    return scattered.GetDirection().Dot(record.Normal) > 0;
}
