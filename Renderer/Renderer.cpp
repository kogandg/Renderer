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

	Camera camera = Camera(16.0 / 9.0, 400, 100, 50);
	tuple<int, int, vector<Color>> pixelInfo = camera.Render(HittableList::Get());

	int imageWidth = get<0>(pixelInfo);
	int imageHeight = get<1>(pixelInfo);
	vector<Color> pixels = get<2>(pixelInfo);
	TGAImage image = TGAImage(imageWidth, imageHeight, 4);
	for (int y = 0; y < imageHeight; y++)
	{
		for (int x = 0; x < imageWidth; x++)
		{
			image.SetPixel(x, y, pixels[x + y * imageWidth]);
		}
	}
	image.WriteTGAFile("out.tga");
}