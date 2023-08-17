#pragma once
#include "Helpers.h"
#include "Color.h"
#include "Hittable.h"
#include "TGAImage.h"
class Camera
{
public:
	double AspectRatio; //Image width over height
	int ImageWidth; //in pixels

	Camera();
	Camera(double aspectRatio, int imageWidth);

	void Render(const Hittable& world, string outFileName);
private:
	int imageHeight; //in pixels
	Vector3 center;
	Vector3 pixel0Center;
	Vector3 pixelDeltaU;
	Vector3 pixelDeltaV;

	void initialize();
	Color rayColor(const Ray& ray, const Hittable& world) const;
};

