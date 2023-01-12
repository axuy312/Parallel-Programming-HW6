__kernel void convolution(__global float *outputImage, __global float *inputImage, __constant float *filter, int imageWidth, int imageHeight, int filterWidth)
{
    int halfFilterWidth = filterWidth / 2;
    float sum = 0;
    int x = get_global_id(0);
    int y = get_global_id(1);

	int imageLTIdx = (y - halfFilterWidth) * imageWidth + x - halfFilterWidth;
	int filterLTIdx = 0;
    for (int r = -halfFilterWidth; r <= halfFilterWidth; r++)
	{
		int tmp = y + r;
		if ( tmp >= 0 && tmp < imageHeight)
		{
			int imageIdx = imageLTIdx;
			int filterIdx = filterLTIdx;
			for (int c = -halfFilterWidth; c <= halfFilterWidth; c++, imageIdx++, filterIdx++)
			{
				tmp = x + c;
				if (tmp >= 0 && tmp < imageWidth)
				{
					float factor = filter[filterIdx];
					if (factor != 0.0)
					{
						sum += (inputImage[imageIdx] * factor);
					}
				}
			}
		}
		imageLTIdx += imageWidth;
		filterLTIdx += filterWidth;
    }
    outputImage[y * imageWidth + x] = sum;
}
