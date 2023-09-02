#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "CImg.h"
using namespace cimg_library;

#include <iostream>
#include <time.h>


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

__global__ void render(float* frameBuffer, int maxX, int maxY)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= maxX) || (y >= maxY))
	{
		return;
	}

	int pixelIndex = y * maxX * 3 + x * 3;
	frameBuffer[pixelIndex + 0] = float(x) / maxX;
	frameBuffer[pixelIndex + 1] = float(y) / maxY;
	frameBuffer[pixelIndex + 2] = 0.2;
}

void hostRender(float* frameBuffer, int maxX, int maxY)
{
	for (int y = 0; y < maxY; y++)
	{
		for (int x = 0; x < maxX; x++)
		{
			int pixelIndex = y * maxX * 3 + x * 3;
			frameBuffer[pixelIndex + 0] = float(x) / maxX;
			frameBuffer[pixelIndex + 1] = float(y) / maxY;
			frameBuffer[pixelIndex + 2] = 0.2;
		}
	}
}

void saveFrameBufferToImage(std::string fileName, float* frameBuffer, int imageX, int imageY)
{
	CImg<unsigned char> image(imageX, imageY, 1, 3, 0);
	for (int y = 0; y < imageY; y++)
	{
		for (int x = 0; x < imageX; x++)
		{
			int pixelIndex = y * 3 * imageX + x * 3;
			unsigned char color[3];
			color[0] = 256 * frameBuffer[pixelIndex];
			color[1] = 256 * frameBuffer[pixelIndex + 1];
			color[2] = 256 * frameBuffer[pixelIndex + 2];
			image.draw_point(x, y, color);
		}
	}
	image.save("fileName");
}

int main()
{
	int* cudaDevices = 0;
	cudaGetDeviceCount(cudaDevices);

	int imageX = 1200;
	int imageY = 600;
	int threadX = 8;
	int threadY = 8;

	int numPixels = imageX * imageY;
	size_t frameBufferSize = 3 * numPixels * sizeof(float);

	float* frameBuffer;
	
	clock_t start;
	clock_t stop;

	if (cudaDevices == 0)
	{
		std::cout << "No cuda devices" << std::endl;

		frameBuffer = new float[frameBufferSize];

		start = clock();
		
		hostRender(frameBuffer, imageX, imageY);

		stop = clock();
	}
	else
	{
		checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

		start = clock();

		dim3 blocks(imageX / threadX + 1, imageY / threadY + 1);
		dim3 threads(threadX, threadY);

		render << <blocks, threads >> > (frameBuffer, imageX, imageY);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		stop = clock();
	}
	
	double timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Rendering took " << timerSeconds << " seconds" << std::endl;

	saveFrameBufferToImage("out.bmp", frameBuffer, imageX, imageY);

	if (cudaDevices == 0)
	{
		delete frameBuffer;
	}
	else
	{
		checkCudaErrors(cudaFree(frameBuffer));
	}
}