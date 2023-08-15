#include <iostream>

#include "Color.h"
#include "TGAImage.h"

using namespace std;

int main()
{
    int imageWidth = 256;
    int imageHeight = 256;

    TGAImage image(imageWidth, imageHeight, 4);
    for (int y = 0; y < imageHeight; y++)
    {
        for (int x = 0; x < imageWidth; x++)
        {
            Color color(double(x) / double(imageWidth - 1), double(y) / double(imageHeight - 1), 0);
            image.SetPixel(x, y, color);
        }
    }
    image.WriteTGAFile("out.tga");
}