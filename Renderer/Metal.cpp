#include "Metal.h"

Metal::Metal(const Color& albedo)
{
    this->albedo = albedo;
}

bool Metal::Scatter(const Ray& rayIn, const HitRecord& record, Color& attenuation, Ray& scattered) const
{
    Vector3 reflected = rayIn.GetDirection().Unit().Reflect(record.Normal);
    scattered = Ray(record.Position, reflected);
    attenuation = albedo;
    return true;
}
