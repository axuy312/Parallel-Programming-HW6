#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
#include <time.h>

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    cl_command_queue clCQueue;
    clCQueue = clCreateCommandQueue(*context, *device, 0, NULL);
	
	//Allocate
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, filterSize, filter, NULL);
    cl_mem d_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, imageSize, inputImage, NULL);
    cl_mem d_outputImage = clCreateBuffer(*context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, imageSize, outputImage, NULL);
	
	cl_kernel convolutionKernel = clCreateKernel(*program, "convolution", NULL);
	
    clSetKernelArg(convolutionKernel, 0, sizeof(cl_mem), (void *)&d_outputImage);
    clSetKernelArg(convolutionKernel, 1, sizeof(cl_mem), (void *)&d_inputImage);
    clSetKernelArg(convolutionKernel, 2, sizeof(cl_mem), (void *)&d_filter);
    clSetKernelArg(convolutionKernel, 3, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(convolutionKernel, 4, sizeof(cl_int), (void *)&imageHeight);
	clSetKernelArg(convolutionKernel, 5, sizeof(cl_int), (void *)&filterWidth);

    size_t localWS[2] = {8, 8};
    size_t globalWS[2] = {imageWidth, imageHeight};
	
	//Run
    clEnqueueNDRangeKernel(clCQueue, convolutionKernel, 2, 0, globalWS, localWS, 0, NULL, NULL);
	
	//Copy D->H
	clEnqueueReadBuffer(clCQueue, d_outputImage, CL_TRUE, 0, imageSize, (void *)outputImage, NULL, NULL, NULL);
}