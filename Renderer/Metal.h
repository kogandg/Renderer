#pragma once
#include "Material.h"
class Metal : public Material
{
private:
	Color albedo;

public:
	Metal(const Color& albedo);

	// Inherited via Material
	bool Scatter(const Ray& rayIn, const HitRecord& record, Color& attenuation, Ray& scattered) const override;
};

