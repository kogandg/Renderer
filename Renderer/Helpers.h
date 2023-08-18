#pragma once

#include <math.h>
#include <limits>
#include <memory>
#include <cstdlib>

using std::shared_ptr;
using std::make_shared;
//using std::sqrt;
using namespace std;

// Constants
const double Infinity = std::numeric_limits<double>::infinity();
const double PI = atan(1.0) * 4;

// Utility Functions
inline double degreesToRadians(double degrees) 
{
	return degrees * PI / 180.0;
}

inline double randomDouble()
{
	return rand() / (RAND_MAX + 1.0);
}

inline double randomDouble(double min, double max)
{
	return min + (max - min) * randomDouble();
}
