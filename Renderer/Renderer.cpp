#include "Helpers.h"

#include <iostream>

#include "Color.h"
#include "TGAImage.h"
#include "Hittable.h"
#include "HittableList.h"
#include "Sphere.h"
#include "Camera.h"
#include "Lambertian.h"
#include "Metal.h"
#include "Dielectric.h"

using namespace std;

int main()
{
	auto groundMaterial = make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
	HittableList::Get().Add(make_shared<Sphere>(Vector3(0, -1000, 0), 1000, groundMaterial));

	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			auto chooseMaterial = randomDouble();
			Vector3 center = Vector3(a + 0.9 * randomDouble(), 0.2, b + 0.9 * randomDouble());
			
			if ((center - Vector3(4, 0.2, 0)).Length() > 0.9)
			{
				shared_ptr<Material> sphereMaterial;
				
				if (chooseMaterial < 0.8)//diffuse
				{
					Color albedo = Color::Random() * Color::Random();
					sphereMaterial = make_shared<Lambertian>(albedo);
				}
				else if (chooseMaterial < 0.95)//metal
				{
					Color albedo = Color::Random(0.5, 1);
					double fuzz = randomDouble(0, 0.5);
					sphereMaterial = make_shared<Metal>(albedo, fuzz);
				}
				else //glass
				{
					sphereMaterial = make_shared<Dielectric>(1.5);
				}
				HittableList::Get().Add(make_shared<Sphere>(center, 0.2, sphereMaterial));
			}
		}
	}
	
	auto material1 = make_shared<Dielectric>(1.5);
	HittableList::Get().Add(make_shared<Sphere>(Vector3(0, 1, 0), 1, material1));

	auto material2 = make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
	HittableList::Get().Add(make_shared<Sphere>(Vector3(-4, 1, 0), 1, material2));

	auto material3 = make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
	HittableList::Get().Add(make_shared<Sphere>(Vector3(4, 1, 0), 1, material3));

	Camera camera = Camera();
	camera.AspectRatio = 16.0 / 9.0;
	camera.ImageWidth = 1200;
	camera.SamplesPerPixel = 500;
	camera.MaxDepth = 50;

	camera.VFOV = 20;
	camera.LookFrom = Vector3(13, 2, 3);
	camera.LookAt = Vector3(0, 0, 0);
	camera.ViewUp = Vector3(0, 1, 0);

	camera.DefocusAngle = 0.6;
	camera.FocusDistance = 10;

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