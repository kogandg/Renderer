#include "Camera.h"

Camera::Camera()
{
	AspectRatio = 1.0;
	ImageWidth = 100;
	
	center = Vector3(0, 0, 0);
}

Camera::Camera(double aspectRatio, int imageWidth)
{
	AspectRatio = aspectRatio;
	ImageWidth = imageWidth;

	center = Vector3(0, 0, 0);
}

void Camera::Render(const Hittable& world, string outFileName)
{
	initialize();
	TGAImage image(ImageWidth, imageHeight, 4);

	for (int y = 0; y < imageHeight; y++)
	{
		for (int x = 0; x < ImageWidth; x++)
		{
			Vector3 pixelCenter = pixel0Center + (pixelDeltaU * x) + (pixelDeltaV * y);
			Vector3 rayDirection = pixelCenter - center;
			Ray ray = Ray(center, rayDirection);

			Color color = rayColor(ray, world);


			image.SetPixel(x, y, color);
		}
	}
	image.WriteTGAFile(outFileName.c_str());
}

void Camera::initialize()
{
	imageHeight = double(ImageWidth) / AspectRatio;
	if (imageHeight < 1)
	{
		imageHeight = 1;
	}

	double focalLength = 1.0;
	double viewportHeight = 2.0;
	double viewportWidth = viewportHeight * (double(ImageWidth) / double(imageHeight));

	//Calc vectors across the horizonal and down the vertical edges of viewport
	Vector3 viewportU = Vector3(viewportWidth, 0, 0);//horizontal
	Vector3 viewportV = Vector3(0, -viewportHeight, 0);//vertical

	//Calc delta vector for pixels across viewport
	pixelDeltaU = viewportU / ImageWidth;
	pixelDeltaV = viewportV / imageHeight;

	//Calc location of upper left pixel
	Vector3 viewportUpperLeft = center - Vector3(0, 0, focalLength) - (viewportU / 2) - (viewportV / 2);
	pixel0Center = viewportUpperLeft + ((pixelDeltaU + pixelDeltaV) * 0.5);
}

Color Camera::rayColor(const Ray& ray, const Hittable& world) const
{
	HitRecord record;
	if (world.Hit(ray, Interval(0, Infinity), record))
	{
		return Color(record.Normal.X + 1, record.Normal.Y + 1, record.Normal.Z + 1) * 0.5;
	}

	Vector3 unitDirection = ray.GetDirection().Unit();
	double a = (unitDirection.Y + 1.0) * 0.5;
	return Color(1.0, 1.0, 1.0) * (1.0 - a) + Color(0.5, 0.7, 1.0) * a;
}
