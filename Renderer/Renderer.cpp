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

using namespace std;

int main()
{
	shared_ptr<Material> groundMaterial = make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
	shared_ptr<Material> centerSphereMaterial = make_shared<Lambertian>(Color(0.7, 0.3, 0.3));
	shared_ptr<Material> leftSphereMaterial = make_shared<Metal>(Color(0.8, 0.8, 0.8));
	shared_ptr<Material> rightSphereMaterial = make_shared<Metal>(Color(0.8, 0.6, 0.2));
	
	HittableList::Get().Add(make_shared<Sphere>(Vector3(0, -100.5, -1), 100, groundMaterial));
	HittableList::Get().Add(make_shared<Sphere>(Vector3(0, 0, -1), 0.5, centerSphereMaterial));
	HittableList::Get().Add(make_shared<Sphere>(Vector3(-1, 0, -1), 0.5, leftSphereMaterial));
	HittableList::Get().Add(make_shared<Sphere>(Vector3(1, 0, -1), 0.5, rightSphereMaterial));
	

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