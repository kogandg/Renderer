#pragma once
#include "cuda_runtime.h"
#include <math.h>

class Color
{
public:
	double R;
	double G;
	double B;

	__host__ __device__ Color();
	__host__ __device__ Color(double r, double g, double b);
	__host__ __device__ Color(const Color& c);

	__host__ __device__ Color LinearToGamma();

	__host__ __device__ Color& operator =(const Color& c);
	__host__ __device__ Color operator*(const Color& obj) const;
	__host__ __device__ Color operator*(double t);
	__host__ __device__ Color operator+(const Color& obj);

	__host__ __device__ Color& operator+=(const Color& rhs);
};