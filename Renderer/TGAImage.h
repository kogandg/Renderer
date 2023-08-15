#pragma once

#include <fstream>
#include <iostream>

#include "Color.h"

using namespace std;

#pragma pack(push,1)
struct TGAHeader {
	char idLength;
	char colorMapType;
	char dataTypeCode;
	short colorMapOrigin;
	short colorMapLength;
	char colorMapDepth;
	short xOrigin;
	short yOrigin;
	short width;
	short height;
	char  bitsPerPixel;
	char  imageDescriptor;
};
#pragma pack(pop)


class TGAImage
{
protected:
	unsigned char* data;
	int width;
	int height;
	int bytesPerPixel;

	bool loadRLEData(std::ifstream& in);
	bool unloadRLEData(std::ofstream& out);
public:
	enum Format {
		GRAYSCALE = 1, RGB = 3, RGBA = 4
	};

	TGAImage();
	TGAImage(int w, int h, int bpp);
	TGAImage(const TGAImage& img);
	~TGAImage();

	bool ReadTGAFile(const char* filename);
	bool WriteTGAFile(const char* filename, bool rle = true);

	bool FlipHorizontally();
	bool FlipVertically();
	bool Scale(int w, int h);

	Color GetPixel(int x, int y);
	bool SetPixel(int x, int y, Color c);
	
	int GetWidth();
	int GetHeight();
	int GetBytesPerPixel();

	unsigned char* Buffer();
	void Clear();

	TGAImage& operator =(const TGAImage& img);
};

