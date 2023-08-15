#include "TGAImage.h"

bool TGAImage::loadRLEData(std::ifstream& in)
{
	unsigned long pixelCount = width * height;
	unsigned long currentPixel = 0;
	unsigned long currentByte = 0;
	Color colorBuffer;

	do
	{
		unsigned char chunkHeader = 0;
		chunkHeader = in.get();
		if (!in.good())
		{
			cerr << "Could not read data" << endl;
			return false;
		}
		if (chunkHeader < 128)
		{
			chunkHeader++;
			for (int i = 0; i < chunkHeader; i++)
			{
				unsigned char raw[4];
				in.read((char*)raw, bytesPerPixel);
				colorBuffer.SetRaw(raw);
				if (!in.good())
				{
					cerr << "Could not read header" << endl;
					return false;
				}
				
				for (int j = 0; j < bytesPerPixel; j++)
				{
					data[currentByte] = raw[j];
					currentByte++;
				}
				currentPixel++;

				if (currentPixel > pixelCount)
				{
					cerr << "Too many pixels to read" << endl;
					return false;
				}
			}
		}
		else
		{
			chunkHeader -= 127;
			unsigned char raw[4];
			in.read((char*)raw, bytesPerPixel);
			colorBuffer.SetRaw(raw);
			if (!in.good())
			{
				cerr << "Could not read header" << endl;
				return false;
			}
			for (int i = 0; i < chunkHeader; i++)
			{
				for (int j = 0; j < bytesPerPixel; j++)
				{
					data[currentByte] = raw[j];
					currentByte++;
				}
				currentPixel++;

				if (currentPixel > pixelCount)
				{
					cerr << "Too many pixels to read" << endl;
					return false;
				}
			}
		}
	} while (currentPixel < pixelCount);

	return true;
}

bool TGAImage::unloadRLEData(std::ofstream& out)
{
	const unsigned char maxChunkLength = 128;
	unsigned long numPixels = width * height;
	unsigned long currentPixel = 0;

	while (currentPixel < numPixels)
	{
		unsigned long chunkStart = currentPixel * bytesPerPixel;
		unsigned long currentByte = currentPixel * bytesPerPixel;
		unsigned char runLength = 1;
		bool raw = true;

		while (currentPixel + runLength < numPixels && runLength < maxChunkLength)
		{
			bool succEq = true;
			for (int i = 0; succEq && i < bytesPerPixel; i++)
			{
				succEq = (data[currentByte + i] == data[currentByte + i + bytesPerPixel]);
			}
			currentByte += bytesPerPixel;

			if (runLength == 1)
			{
				raw = !succEq;
			}
			if (raw && succEq)
			{
				runLength--;
				break;
			}
			if (!raw && !succEq)
			{
				break;
			}
			runLength++;
		}

		currentPixel += runLength;

		out.put(raw ? runLength - 1 : runLength + 127);
		if (!out.good())
		{
			cerr << "Could not dump TGA file" << endl;
			return false;
		}

		out.write((char*)(data + chunkStart), (raw ? runLength * bytesPerPixel : bytesPerPixel));
		if (!out.good())
		{
			cerr << "Could not dump TGA file" << endl;
			return false;
		}
	}

	return true;
}

TGAImage::TGAImage()
{
	data = NULL;
	width = 0;
	height = 0;
	bytesPerPixel = 0;
}

TGAImage::TGAImage(int w, int h, int bpp)
{
	width = w;
	height = h;
	bytesPerPixel = bpp;

	unsigned long numBytes = width * height * bytesPerPixel;
	data = new unsigned char[numBytes];
	memset(data, 0, numBytes);
}

TGAImage::TGAImage(const TGAImage& img)
{
	width = img.width;
	height = img.height;
	bytesPerPixel = img.bytesPerPixel;
	unsigned long numBytes = width * height * bytesPerPixel;
	data = new unsigned char[numBytes];
	memcpy(data, img.data, numBytes);
}

TGAImage::~TGAImage()
{
	if (data)
	{
		delete[] data;
	}
}

bool TGAImage::ReadTGAFile(const char* filename)
{
	if (data)
	{
		delete[] data;
	}
	data = NULL;

	ifstream in;
	in.open(filename, ios::binary);
	if (!in.is_open())
	{
		cerr << "Could not open file " << filename << endl;
		in.close();
		return false;
	}

	TGAHeader header;
	in.read((char*)&header, sizeof(header));

	if (!in.good())
	{
		cerr << "Could not read header" << endl;
		in.close();
		return false;
	}

	width = header.width;
	height = header.height;
	bytesPerPixel = header.bitsPerPixel >> 3;

	if (width <= 0 || height <= 0 || (bytesPerPixel != GRAYSCALE && bytesPerPixel != RGB && bytesPerPixel != RGBA))
	{
		cerr << "Bad width/height/bytesPerPixel value" << endl;
		in.close();
		return false;
	}

	unsigned long numBytes = bytesPerPixel * width * height;
	data = new unsigned char[numBytes];
	if (header.dataTypeCode == 2 || header.dataTypeCode == 3)
	{
		in.read((char*)data, numBytes);
		if (!in.good())
		{
			cerr << "Could not read data" << endl;
			in.close();
			return false;
		}
	}
	else if (header.dataTypeCode == 10 || header.dataTypeCode == 11)
	{
		if (!loadRLEData(in))
		{
			cerr << "Could not read data" << endl;
			in.close();
			return false;
		}
	}
	else
	{
		cerr << "Uknown file format " << (int)header.dataTypeCode << endl;
		in.close();
		return false;
	}

	if (!(header.imageDescriptor & 0x20))
	{
		FlipVertically();
	}
	if (header.imageDescriptor & 0x10)
	{
		FlipHorizontally();
	}

	clog << width << "x" << height << "/" << bytesPerPixel * 8 << endl;
	in.close();
	return true;
}

bool TGAImage::WriteTGAFile(const char* filename, bool rle)
{
	unsigned char developerAreaRef[4] = { 0, 0, 0, 0 };
	unsigned char extensionAreaRef[4] = { 0, 0, 0, 0 };
	string footer = "TRUEVISION-XFILE.\0";

	ofstream out;
	out.open(filename, ios::binary);
	if (!out.is_open())
	{
		cerr << "Could not open file " << filename << endl;
		out.close();
		return false;
	}

	TGAHeader header;
	memset((void*)&header, 0, sizeof(header));
	header.bitsPerPixel = bytesPerPixel << 3;
	header.width = width;
	header.height = height;
	header.dataTypeCode = (bytesPerPixel == GRAYSCALE ? (rle ? 11 : 3) : (rle ? 10 : 2));
	header.imageDescriptor = 0x20;

	out.write((char*)&header, sizeof(header));
	if (!out.good())
	{
		cerr << "Could not dump TGA file" << endl;
		out.close();
		return false;
	}

	if (!rle)
	{
		out.write((char*)data, width * height * bytesPerPixel);
		if (!out.good())
		{
			cerr << "Could not unload raw data" << endl;
			out.close();
			return false;
		}
	}
	else if (!unloadRLEData(out))
	{
		cerr << "Could not unload RLE data" << endl;
		out.close();
		return false;
	}

	out.write((char*)developerAreaRef, sizeof(developerAreaRef));
	if (!out.good())
	{
		cerr << "Could not dump TGA file" << endl;
		out.close();
		return false;
	}

	out.write((char*)extensionAreaRef, sizeof(extensionAreaRef));
	if (!out.good())
	{
		cerr << "Could not dump TGA file" << endl;
		out.close();
		return false;
	}

	out.write(footer.c_str(), sizeof(footer.c_str()));
	if (!out.good())
	{
		cerr << "Could not dump TGA file" << endl;
		out.close();
		return false;
	}

	out.close();
	return true;
}

bool TGAImage::FlipHorizontally()
{
	if (!data)
	{
		return false;
	}

	int half = width / 2;
	for (int i = 0; i < half; i++)
	{
		for (int j = 0; j < height; j++)
		{
			Color color1 = GetPixel(i, j);
			Color color2 = GetPixel(width - 1 - i, j);
			SetPixel(i, j, color2);
			SetPixel(width - 1 - i, j, color1);
		}
	}
	return true;
}

bool TGAImage::FlipVertically()
{
	if (!data)
	{
		return false;
	}

	unsigned long bytesPerLine = width * bytesPerPixel;
	unsigned char* line = new unsigned char[bytesPerLine];

	int half = height / 2;
	for (int i = 0; i < half; i++)
	{
		unsigned long line1 = i * bytesPerLine;
		unsigned long line2 = (height - 1 - i) * bytesPerLine;
		memmove((void*)line, (void*)(data + line1), bytesPerLine);
		memmove((void*)(data + line1), (void*)(data + line2), bytesPerLine);
		memmove((void*)(data + line2), (void*)line, bytesPerLine);
	}

	delete[] line;
	return true;
}

bool TGAImage::Scale(int w, int h)
{
	if (w <= 0 || h <= 0 || !data)
	{
		return false;
	}

	unsigned char* tempData = new unsigned char[w * h * bytesPerPixel];
	int nScanLine = 0;
	int oScanLine = 0;
	int errY = 0;
	unsigned long nLineBytes = w * bytesPerPixel;
	unsigned long oLineBytes = width * bytesPerPixel;

	for (int i = 0; i < height; i++)
	{
		int errX = width - w;
		int nX = -bytesPerPixel;
		int oX = -bytesPerPixel;
		
		for (int j = 0; j < width; j++)
		{
			oX += bytesPerPixel;
			errX += w;
			while (errX >= width)
			{
				errX -= width;
				nX += bytesPerPixel;
				memcpy(tempData + nScanLine + nX, data + oScanLine + oX, bytesPerPixel);
			}
		}

		errY += h;
		oScanLine += oLineBytes;
		while (errY >= height)
		{
			if (errY >= height / 2) 
			{
				memcpy(tempData + nScanLine + nLineBytes, tempData + nScanLine, nLineBytes);
			}
			errY -= height;
			nScanLine += nLineBytes;
		}
	}

	delete[] data;
	data = tempData;
	width = w;
	height = h;
	return true;
}

Color TGAImage::GetPixel(int x, int y)
{
	if (!data || x < 0 || y < 0 || x >= width || y >= height)
	{
		return Color();
	}
	return Color(data + (x + y * width) * bytesPerPixel, bytesPerPixel);
}

bool TGAImage::SetPixel(int x, int y, Color c)
{
	if (!data || x < 0 || y < 0 || x >= width || y >= height)
	{
		return false;
	}
	memcpy(data + (x + y * width) * bytesPerPixel, c.GetRaw(), bytesPerPixel);
	return true;
}

int TGAImage::GetWidth()
{
	return width;
}

int TGAImage::GetHeight()
{
	return height;
}

int TGAImage::GetBytesPerPixel()
{
	return bytesPerPixel;
}

unsigned char* TGAImage::Buffer()
{
	return data;
}

void TGAImage::Clear()
{
	memset((void*)data, 0, width * height * bytesPerPixel);
}

TGAImage& TGAImage::operator=(const TGAImage& img)
{
	if (this != &img)
	{
		if (data)
		{
			delete[] data;
		}

		width = img.width;
		height = img.height;
		bytesPerPixel = img.bytesPerPixel;
		
		unsigned long numBytes = width * height * bytesPerPixel;
		data = new unsigned char[numBytes];
		memcpy(data, img.data, numBytes);
	}
	return *this;
}
