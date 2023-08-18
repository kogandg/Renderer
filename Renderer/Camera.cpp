#include "Camera.h"

Camera::Camera()
{
	AspectRatio = 1.0;
	ImageWidth = 100;
	SamplesPerPixel = 1;

	center = Vector3(0, 0, 0);
}

Camera::Camera(double aspectRatio, int imageWidth, int samples)
{
	AspectRatio = aspectRatio;
	ImageWidth = imageWidth;
	SamplesPerPixel = samples;

	center = Vector3(0, 0, 0);
}

tuple<int, int, vector<Color>> Camera::Render(const Hittable& world)
{
	initialize();

	vector<Color> pixels = vector<Color>(ImageWidth * imageHeight);
	for (int y = 0; y < imageHeight; y++)
	{
		for (int x = 0; x < ImageWidth; x++)
		{
			Vector3 pixelCenter = pixel0Center + (pixelDeltaU * x) + (pixelDeltaV * y);
			Vector3 rayDirection = pixelCenter - center;

			Color pixelColor = Color(0, 0, 0);
			for (int sample = 0; sample < SamplesPerPixel; sample++)
			{
				Ray ray = getRay(x, y);
				pixelColor += rayColor(ray, world);
			}
			double scale = 1.0 / SamplesPerPixel;
			pixelColor = pixelColor * scale;

			//Ray ray = Ray(center, rayDirection);
			//Color color = rayColor(ray, world);
			
			pixels[x + y * ImageWidth] = pixelColor;
		}
	}

	return make_tuple(ImageWidth, imageHeight, pixels);
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

Ray Camera::getRay(int x, int y) const
{
	Vector3 pixelCenter = pixel0Center + (pixelDeltaU * x) + (pixelDeltaV * y);
	Vector3 pixelSample = pixelCenter + pixelSampleSquare();

	Vector3 rayOrigin = center;
	Vector3 rayDirection = pixelSample - rayOrigin;
	return Ray(rayOrigin, rayDirection);
}

Vector3 Camera::pixelSampleSquare() const
{
	double px = -0.5 + randomDouble();
	double py = -0.5 + randomDouble();
	return (pixelDeltaU * px) + (pixelDeltaV * py);
}
