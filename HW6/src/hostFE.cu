#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "helper.h"
extern "C"{
#include "hostFE.h"
}

__global__ void convKernel(float *outputImage, float *inputImage, float *filter, int imageWidth, int imageHeight, int filterWidth){
    
    int halfFilterWidth = filterWidth / 2;
    float sum = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;

	int imageLTIdx = (y - halfFilterWidth) * imageWidth + x - halfFilterWidth;
	int filterLTIdx = 0;
    for (int r = -halfFilterWidth; r <= halfFilterWidth; r++)
	{
		if (y + r >= 0 && y + r < imageHeight)
		{
			int imageIdx = imageLTIdx;
			int filterIdx = filterLTIdx;
			for (int c = -halfFilterWidth; c <= halfFilterWidth; c++, imageIdx++, filterIdx++)
			{
				if (x + c >= 0 && x + c < imageWidth)
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

extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    float *d_filter;
    float *d_inputImage;
    float *d_outputImage;
	
	//Allocate
    cudaMalloc(&d_filter, filterSize);
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

	//Copy H->D
    cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    int numThreads = imageWidth;
    int numBlocks = imageHeight;
    convKernel<<<numBlocks, numThreads>>>(d_outputImage, d_inputImage, d_filter, imageWidth, imageHeight, filterWidth);

	//Copy D->H
    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_filter);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}