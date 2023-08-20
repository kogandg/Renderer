#pragma once
#include "Material.h"
class Dielectric : public Material
{
private:
	double indexOfRefraction;

	static double reflectance(double cosine, double refractionRatio);

public:
	Dielectric(double indexOfRefraction);

	// Inherited via Material
	bool Scatter(const Ray& rayIn, const HitRecord& record, Color& attenuation, Ray& scattered) const override;

};

