#include <iostream>
#include <math.h>

#include "Color.h"
#include "TGAImage.h"
#include "Vector3.h"
#include "Ray.h"

using namespace std;

double hitSphere(const Vector3& center, double radius, const Ray& r)
{
	Vector3 oc = r.GetOrigin() - center;
	//quaradic equation
	double a = r.GetDirection().Dot(r.GetDirection());
	double b = oc.Dot(r.GetDirection()) * 2.0;
	double c = oc.Dot(oc) - (radius * radius);
	double discriminant = (b * b) - (4 * a * c);

	if (discriminant < 0)
	{
		return -1.0;
	}
	else
	{
		return (-b - sqrt(discriminant)) / (2.0 * a);
	}
}

Color rayColor(const Ray& r) {
	double t = hitSphere(Vector3(0, 0, -1), 0.5, r);
	if (t > 0.0)
	{
		Vector3 normal = (r.At(t) - Vector3(0, 0, -1)).Unit();
		return Color(normal.X + 1, normal.Y + 1, normal.Z + 1) * 0.5;
	}

	Vector3 unitDirection = r.GetDirection().Unit();
	double a = (unitDirection.Y + 1.0) * 0.5;
	return Color(1.0, 1.0, 1.0) * (1.0 - a) + Color(0.5, 0.7, 1.0) * a;
}

int main()
{
	//Image
	double aspectRatio = 16.0 / 9.0;
	int imageWidth = 400;

	//Calc height using aspect ratio and ensure it is at least 1
	int imageHeight = double(imageWidth) / aspectRatio;
	if (imageHeight < 1)
	{
		imageHeight = 1;
	}

	TGAImage image(imageWidth, imageHeight, 4);

	//Camera
	double focalLength = 1.0;
	double viewportHeight = 2.0;
	double viewportWidth = viewportHeight * (double(imageWidth) / double(imageHeight));
	Vector3 cameraCenter = Vector3(0, 0, 0);

	//Calc vectors across the horizonal and down the vertical edges of viewport
	Vector3 viewportU = Vector3(viewportWidth, 0, 0);//horizontal
	Vector3 viewportV = Vector3(0, -viewportHeight, 0);//vertical

	//Calc delta vector for pixels across viewport
	Vector3 pixelDeltaU = viewportU / imageWidth;
	Vector3 pixelDeltaV = viewportV / imageHeight;

	//Calc location of upper left pixel
	Vector3 viewportUpperLeft = cameraCenter - Vector3(0, 0, focalLength) - (viewportU / 2) - (viewportV / 2);
	Vector3 pixel0Center = viewportUpperLeft + ((pixelDeltaU + pixelDeltaV) * 0.5);

	//Render
	for (int y = 0; y < imageHeight; y++)
	{
		//clog << "\x1b[2K";
		//clog << "\rScanlines remaining: " << (imageHeight - y) << ' ' << flush;

		for (int x = 0; x < imageWidth; x++)
		{
			Vector3 pixelCenter = pixel0Center + (pixelDeltaU * x) + (pixelDeltaV * y);
			Vector3 rayDirection = pixelCenter - cameraCenter;
			Ray ray = Ray(cameraCenter, rayDirection);

			//Color color = Color(double(x) / double(imageWidth - 1), double(y) / double(imageHeight - 1), 0);
			Color color = rayColor(ray);


			image.SetPixel(x, y, color);
		}
	}
	//clog << "\x1b[2K";
	//clog << "\rDone";
	image.WriteTGAFile("out.tga");
}