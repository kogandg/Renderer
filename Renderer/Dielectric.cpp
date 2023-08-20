#include "Dielectric.h"

double Dielectric::reflectance(double cosine, double refractionRatio)
{
    double r0 = (1 - refractionRatio) / (1 + refractionRatio);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

Dielectric::Dielectric(double indexOfRefraction)
{
    this->indexOfRefraction = indexOfRefraction;
}

bool Dielectric::Scatter(const Ray& rayIn, const HitRecord& record, Color& attenuation, Ray& scattered) const
{
    attenuation = Color(1, 1, 1);
    double refractionRatio = record.FrontFace ? (1 / indexOfRefraction) : indexOfRefraction;

    Vector3 unitDirection = rayIn.GetDirection().Unit();

    double cosTheta = fmin((-unitDirection.Dot(record.Normal)), 1.0);
    double sinTheta = sqrt(1.0-cosTheta*cosTheta);

    bool cannotRefract = refractionRatio * sinTheta > 1.0;
    Vector3 direction;

    if (cannotRefract || reflectance(cosTheta, refractionRatio) > randomDouble())
    {
        direction = unitDirection.Reflect(record.Normal);
    }
    else
    {
        direction = unitDirection.Refract(record.Normal, refractionRatio);
    }

    scattered = Ray(record.Position, direction);
    return true;
}
