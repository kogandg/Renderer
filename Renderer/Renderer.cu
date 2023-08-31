//#include "Helpers.h"
//
//#include <iostream>
//
//#include "Color.h"
//#include "TGAImage.h"
//#include "Hittable.h"
//#include "HittableList.h"
//#include "Sphere.h"
//#include "Camera.h"
//#include "Lambertian.h"
//#include "Metal.h"
//#include "Dielectric.h"
//
//using namespace std;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main()
{
	thrust::device_vector<int> vectorTest;
	/*shared_ptr<Material> groundMaterial = make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
	shared_ptr<Material> centerSphereMaterial = make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
	shared_ptr<Material> leftSphereMaterial = make_shared<Dielectric>(1.5);
	shared_ptr<Material> rightSphereMaterial = make_shared<Metal>(Color(0.8, 0.6, 0.2), 0.0);
	
	HittableList::Get().Add(make_shared<Sphere>(Vector3(0, -100.5, -1), 100, groundMaterial));
	HittableList::Get().Add(make_shared<Sphere>(Vector3(0, 0, -1), 0.5, centerSphereMaterial));

	HittableList::Get().Add(make_shared<Sphere>(Vector3(-1, 0, -1), 0.5, leftSphereMaterial));
	HittableList::Get().Add(make_shared<Sphere>(Vector3(-1, 0, -1), -0.4, leftSphereMaterial));

	HittableList::Get().Add(make_shared<Sphere>(Vector3(1, 0, -1), 0.5, rightSphereMaterial));
	
	Camera camera = Camera();
	camera.AspectRatio = 16.0 / 9.0;
	camera.ImageWidth = 400;
	camera.SamplesPerPixel = 100;
	camera.MaxDepth = 50;

	camera.VFOV = 20;
	camera.LookFrom = Vector3(-2, 2, 1);
	camera.LookAt = Vector3(0, 0, -1);
	camera.ViewUp = Vector3(0, 1, 0);

	camera.DefocusAngle = 10;
	camera.FocusDistance = 3.4;

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
	image.WriteTGAFile("out.tga");*/
}