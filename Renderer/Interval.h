#pragma once
#include "Helpers.h";
class Interval
{
public:
	double Min;
	double Max;

	Interval();
	Interval(double min, double max);

	bool Contains(double x) const;
	bool Surrounds(double x) const;

	static const Interval Empty;
	static const Interval Universe;
};

const static Interval Empty(+Infinity, -Infinity);
const static Interval Universe(-Infinity, +Infinity);