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

	double VFOV; //vertical FOV in degrees
	Vector3 LookFrom;
	Vector3 LookAt;
	Vector3 ViewUp; //camera relative up direction

	double DefocusAngle; //variation angle of rays through each pixel
	double FocusDistance; //distance from camera lookfrom to plane of perfect focus

	Camera();
	Camera(double aspectRatio, int imageWidth, int samples, int maxDepth, double vfov, Vector3 lookFrom, Vector3 lookAt, Vector3 viewUp, double defocusAngle, double focusDistance);

	tuple<int, int, vector<Color>> Render(const Hittable& world);
private:
	int imageHeight; //in pixels
	Vector3 center;
	Vector3 pixel0Center;
	Vector3 pixelDeltaU;
	Vector3 pixelDeltaV;

	//Camera fram basis vectors
	Vector3 u;
	Vector3 v;
	Vector3 w;

	Vector3 defocusDiskU; //defocus disk horizontal radius
	Vector3 defocusDiskV; //defocus disk vertical radius

	void initialize();
	Color rayColor(const Ray& ray, int depth, const Hittable& world) const;

	Ray getRay(int x, int y) const;
	Vector3 pixelSampleSquare() const;
	Vector3 defocusDiskSample() const;
};

