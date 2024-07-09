#include "common.h"
#include "timer.h"
#define COARSE_FACTOR 4

__global__ void histogram_private_kernel(unsigned char *image, unsigned int *bins, unsigned int width, unsigned int height)
{
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ unsigned int b_s[NUM_BINS];
  if (threadIdx.x < NUM_BINS)
  {
    b_s[threadIdx.x] = 0;
  }
  __syncthreads();
  if (i < width * height)
  {
    unsigned int b = image[i];
    atomicAdd(&b_s[b], 1);
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    for (int j = 0; j < NUM_BINS; ++j)
    {
      if (b_s[j] > 0)
      {
        atomicAdd(&bins[j], b_s[j]);
      }
    }
  }
}

void histogram_gpu_private(unsigned char *image_d, unsigned int *bins_d, unsigned int width, unsigned int height)
{
  unsigned int numThreadsPerBlock = 1024;
  unsigned int numBlocks = (height * width + numThreadsPerBlock - 1) / numThreadsPerBlock;
  histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
}

__global__ void histogram_private_coarse_kernel(unsigned char *image, unsigned int *bins, unsigned int width, unsigned int height)
{
  unsigned int i_start = blockDim.x * blockIdx.x * COARSE_FACTOR + threadIdx.x;
  __shared__ unsigned int b_s[NUM_BINS];
  if (threadIdx.x < NUM_BINS)
  {
    b_s[threadIdx.x] = 0;
  }
  __syncthreads();
  if (i_start < height * width)
  {
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
    {
      unsigned int i = i_start + c * blockDim.x;
      if (i < height * width)
      {
        unsigned int b = image[i];
        atomicAdd(&b_s[b], 1);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    for (int j = 0; j < NUM_BINS; ++j)
    {
      if (b_s[j] > 0)
      {
        atomicAdd(&bins[j], b_s[j]);
      }
    }
  }
}

void histogram_gpu_private_coarse(unsigned char *image_d, unsigned int *bins_d, unsigned int width, unsigned int height)
{
  unsigned int numThreadsPerBlock = 1024;
  unsigned int numBlocks = (height * width + numThreadsPerBlock * COARSE_FACTOR - 1) / (numThreadsPerBlock * COARSE_FACTOR);
  histogram_private_coarse_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
}