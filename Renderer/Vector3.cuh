#pragma once

#include "cuda_runtime.h"
#include <math.h>

class Vector3
{
public:
	double X;
	double Y;
	double Z;

	__host__ __device__ Vector3();
	__host__ __device__ Vector3(double x, double y, double z);
	__host__ __device__ Vector3(const Vector3& v);

	__host__ __device__ double Length();
	__host__ __device__ double LengthSquared();

	__host__ __device__ bool NearZero() const;

	__host__ __device__ double Dot(const Vector3& v);
	__host__ __device__ Vector3 Cross(const Vector3& v);

	__host__ __device__ Vector3 Reflect(const Vector3& normal);
	__host__ __device__ Vector3 Refract(const Vector3& normal, double etaiOverEtat);

	__host__ __device__ Vector3 Unit();

	__host__ __device__ Vector3 operator-() const;

	__host__ __device__ Vector3& operator+=(const Vector3& rhs);

	__host__ __device__ Vector3& operator*=(double t);
	__host__ __device__ Vector3& operator/=(double t);

	__host__ __device__ Vector3 operator+(const Vector3& obj) const;
	__host__ __device__ Vector3 operator-(const Vector3& obj) const;
	__host__ __device__ Vector3 operator*(const Vector3& obj) const;

	__host__ __device__ Vector3 operator*(double t) const;
	__host__ __device__ Vector3 operator/(double t) const;
};