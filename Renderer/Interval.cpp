#include "Interval.h"

Interval::Interval()
{
	Min = +Infinity;
	Max = -Infinity;
}

Interval::Interval(double min, double max)
{
	Min = min;
	Max = max;
}

bool Interval::Contains(double x) const
{
	return Min <= x && Max >= x;
}

bool Interval::Surrounds(double x) const
{
	return Min < x && Max > x;
}