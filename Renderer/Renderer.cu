#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "CImg.h"
using namespace cimg_library;

#include <iostream>
#include <time.h>

#include "Color.cuh"
#include "Ray.cuh"

#pragma region Utils

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

void saveFrameBufferToImage(std::string fileName, Color* frameBuffer, int imageX, int imageY)
{
	CImg<unsigned char> image(imageX, imageY, 1, 3, 0);
	for (int y = 0; y < imageY; y++)
	{
		for (int x = 0; x < imageX; x++)
		{
			int pixelIndex = y * imageX + x;
			unsigned char color[3];
			color[0] = 255.999 * frameBuffer[pixelIndex].R;
			color[1] = 255.999 * frameBuffer[pixelIndex].G;
			color[2] = 255.999 * frameBuffer[pixelIndex].B;
			image.draw_point(x, y, color);
		}
	}
	image.save(fileName.c_str());
}

#pragma endregion

__host__ __device__ bool hitSphere(const Vector3& center, float radius, const Ray& ray) {
	Vector3 oc = ray.GetOrigin() - center;
	float a = ray.GetDirection().Dot(ray.GetDirection());
	float b = 2.0f * oc.Dot(ray.GetDirection());
	float c = oc.Dot(oc) - radius * radius;
	float discriminant = b * b - 4.0f * a * c;
	return (discriminant > 0.0f);
}

__host__ __device__ Color rayColor(const Ray& ray)
{
	if (hitSphere(Vector3(0, 0, -1), 0.5, ray))
	{
		return Color(1, 0, 0);
	}
	Vector3 unitDirection = ray.GetDirection().Unit();
	float t = 0.5f * (unitDirection.Y + 1.0f);
	return Color(1.0, 1.0, 1.0) * (1.0 - t) + Color(0.5, 0.7, 1.0) * t;
}

__global__ void render(Color* frameBuffer, int maxX, int maxY, Vector3 pixel0Center, Vector3 pixelDeltaU, Vector3 pixelDeltaV, Vector3 cameraCenter)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= maxX) || (y >= maxY))
	{
		return;
	}

	Vector3 pixelCenter = pixel0Center + (pixelDeltaU * x) + (pixelDeltaV * y);
	Vector3 rayDirection = pixelCenter - cameraCenter;
	Ray ray = Ray(cameraCenter, rayDirection);

	Color color = rayColor(ray);

	int pixelIndex = y * maxX + x;
	frameBuffer[pixelIndex] = color;
}

void hostRender(Color* frameBuffer, int maxX, int maxY, Vector3 pixel0Center, Vector3 pixelDeltaU, Vector3 pixelDeltaV, Vector3 cameraCenter)
{
	for (int y = 0; y < maxY; y++)
	{
		for (int x = 0; x < maxX; x++)
		{
			Vector3 pixelCenter = pixel0Center + (pixelDeltaU * x) + (pixelDeltaV * y);
			Vector3 rayDirection = pixelCenter - cameraCenter;
			Ray ray = Ray(cameraCenter, rayDirection);

			Color color = rayColor(ray);

			int pixelIndex = y * maxX + x;
			frameBuffer[pixelIndex] = color;
		}
	}
}


int main()
{
	int cudaDevices = 0;
	cudaGetDeviceCount(&cudaDevices);

	int imageWidth = 1200;

	double aspectRatio = 16.0 / 9.0;
	int imageHeight = (double)(imageWidth) / aspectRatio;
	if (imageHeight < 1)
	{
		imageHeight = 1;
	}


	//Camera info
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
	Vector3 viewportUpperLeft = cameraCenter - Vector3(0, 0, focalLength) - viewportU / 2 - viewportV / 2;
	Vector3 pixel0Center = viewportUpperLeft + (pixelDeltaU + pixelDeltaV) * 0.5;


	int numPixels = imageWidth * imageHeight;
	Color* frameBuffer;
	
	clock_t start;
	clock_t stop;

	//cudaDevices = 0;
	if (cudaDevices == 0)
	{
		std::cout << "No cuda devices" << std::endl;

		frameBuffer = new Color[numPixels];

		start = clock();
		
		hostRender(frameBuffer, imageWidth, imageHeight, pixel0Center, pixelDeltaU, pixelDeltaV, cameraCenter);

		stop = clock();
	}
	else
	{
		size_t frameBufferSize = numPixels * sizeof(Color);
		checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

		int threadX = 8;
		int threadY = 8;

		start = clock();

		dim3 blocks(imageWidth / threadX + 1, imageHeight / threadY + 1);
		dim3 threads(threadX, threadY);

		render << <blocks, threads >> > (frameBuffer, imageWidth, imageHeight, pixel0Center, pixelDeltaU, pixelDeltaV, cameraCenter);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		stop = clock();
	}
	
	double timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Rendering took " << timerSeconds << " seconds" << std::endl;

	saveFrameBufferToImage("out.bmp", frameBuffer, imageWidth, imageHeight);

	if (cudaDevices == 0)
	{
		delete frameBuffer;
	}
	else
	{
		checkCudaErrors(cudaFree(frameBuffer));
	}
}