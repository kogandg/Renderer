#pragma once
#include "cuda_runtime.h"

#include "Vector3.cuh"

class Ray
{
private:
	Vector3 origin;
	Vector3 direction;
public:
	__host__ __device__ Ray();
	__host__ __device__ Ray(const Vector3& origin, const Vector3& direction);

	__host__ __device__ Vector3 At(double t) const;

	__host__ __device__ Vector3 GetOrigin() const;
	__host__ __device__ Vector3 GetDirection() const;
};