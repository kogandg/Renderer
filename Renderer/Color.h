#pragma once
class Color
{
private:
	int bytesPerColor;
	union
	{
		struct
		{
			unsigned char b;
			unsigned char g;
			unsigned char r;
			unsigned char a;
		};
		unsigned char raw[4];
		unsigned int value;
	};

	void matchRaws();
	void matchReals();

public:
	Color();
	Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
	Color(double r, double g, double b, double a);
	Color(double r, double g, double b);
	Color(int val, int bpc);
	Color(const Color& c);
	Color(const unsigned char* p, int bpc);

	double R;
	double G;
	double B;
	double A;

	unsigned char* GetRaw();
	void SetRaw(unsigned char raw[]);

	Color& operator =(const Color &c);
	Color operator*(double t);
	Color operator+(const Color& obj);

	Color& operator+=(const Color& rhs);
};

