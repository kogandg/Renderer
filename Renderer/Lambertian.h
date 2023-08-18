#pragma once
#include "Material.h"
class Lambertian : public Material
{
private:
	Color albedo;

public:
	Lambertian(const Color& albedo);

	// Inherited via Material
	bool Scatter(const Ray& rayIn, const HitRecord& record, Color& attenuation, Ray& scattered) const override;
};

