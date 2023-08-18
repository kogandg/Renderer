#include "Lambertian.h"

Lambertian::Lambertian(const Color& albedo)
{
    this->albedo = albedo;
}

bool Lambertian::Scatter(const Ray& rayIn, const HitRecord& record, Color& attenuation, Ray& scattered) const
{
    Vector3 scatterDirection = record.Normal + Vector3::RandomUnitVector();
    if (scatterDirection.NearZero())
    {
        scatterDirection = record.Normal;
    }
    scattered = Ray(record.Position, scatterDirection);

    attenuation = albedo;

    return true;
}
