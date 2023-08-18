#pragma once
#include "Hittable.h"
#include "Ray.h"
#include "Color.h"

class Material
{
public:
	virtual ~Material() = default;

	virtual bool Scatter(const Ray& rayIn, const HitRecord& record, Color& attenuation, Ray& scattered) const = 0;
};

