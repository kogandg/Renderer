#include "Helpers.h"

#include <iostream>

#include "Color.h"
#include "TGAImage.h"
#include "Hittable.h"
#include "HittableList.h"
#include "Sphere.h"
#include "Camera.h"

using namespace std;

Color rayColor(const Ray& ray) {
	HitRecord record;
	if (HittableList::Get().Hit(ray, Interval(0, Infinity), record))
	{
		return Color(record.Normal.X + 1, record.Normal.Y + 1, record.Normal.Z + 1) * 0.5;
	}

	Vector3 unitDirection = ray.GetDirection().Unit();
	double a = (unitDirection.Y + 1.0) * 0.5;
	return Color(1.0, 1.0, 1.0) * (1.0 - a) + Color(0.5, 0.7, 1.0) * a;
}

int main()
{

	//World
	HittableList::Get().Add(make_shared<Sphere>(Vector3(0, 0, -1), 0.5));
	HittableList::Get().Add(make_shared<Sphere>(Vector3(0, -100.5, -1), 100));

	Camera camera = Camera(16.0/9.0, 400);
	camera.Render(HittableList::Get(), "out.tga");
}