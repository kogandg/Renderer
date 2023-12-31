#pragma once
#include "Material.h"
class Metal : public Material
{
private:
	Color albedo;
	double fuzz;

public:
	Metal(const Color& albedo, double fuzz);

	// Inherited via Material
	bool Scatter(const Ray& rayIn, const HitRecord& record, Color& attenuation, Ray& scattered) const override;
};

