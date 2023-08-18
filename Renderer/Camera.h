#pragma once
#include "Helpers.h"
#include "Color.h"
#include "Hittable.h"
#include "Vector3.h"
#include "Ray.h"
#include "Interval.h"
#include "Material.h"
#include <iostream>
#include <tuple>
#include <vector>

class Camera
{
public:
	double AspectRatio; //Image width over height
	int ImageWidth; //in pixels
	int SamplesPerPixel;
	int MaxDepth;

	Camera();
	Camera(double aspectRatio, int imageWidth, int samples, int maxDepth);

	tuple<int, int, vector<Color>> Render(const Hittable& world);
private:
	int imageHeight; //in pixels
	Vector3 center;
	Vector3 pixel0Center;
	Vector3 pixelDeltaU;
	Vector3 pixelDeltaV;

	void initialize();
	Color rayColor(const Ray& ray, int depth, const Hittable& world) const;

	Ray getRay(int x, int y) const;
	Vector3 pixelSampleSquare() const;
};

