
#include "common.h"

#include "timer.h"

#define BLOCK_DIM 1024

__global__ void scan_kernel(float *input, float *output, float *partialSums, unsigned int N)
{
    unsigned int segment = 2 * blockIdx.x * blockDim.x;
    __shared__ float buffer_s[2 * BLOCK_DIM];

    if ((segment + threadIdx.x + BLOCK_DIM) < N)
    {
        buffer_s[threadIdx.x] = input[segment + threadIdx.x];
        buffer_s[threadIdx.x + BLOCK_DIM] = input[segment + threadIdx.x + BLOCK_DIM];
    }
    else if ((segment + threadIdx.x) < N)
    {
        buffer_s[threadIdx.x] = input[segment + threadIdx.x];
        buffer_s[threadIdx.x + BLOCK_DIM] = 0;
    }
    else
    {
        buffer_s[threadIdx.x] = 0;
        buffer_s[threadIdx.x + BLOCK_DIM] = 0;
    }
    __syncthreads();

    // First tree
    for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2)
    {
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;
        if (i < 2 * BLOCK_DIM)
        {
            buffer_s[i] += buffer_s[i - stride];
        }
        __syncthreads();
    }

    // Store partial sum
    if (threadIdx.x == 0)
    {
        partialSums[blockIdx.x] = buffer_s[2 * BLOCK_DIM - 1];
        buffer_s[2 * BLOCK_DIM - 1] = 0;
    }
    __syncthreads();

    // Second tree
    for (unsigned int stride = BLOCK_DIM; stride >= 1; stride /= 2)
    {
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;
        if (i < 2 * BLOCK_DIM)
        {
            float temp = buffer_s[i];
            buffer_s[i] += buffer_s[i - stride];
            buffer_s[i - stride] = temp;
        }
        __syncthreads();
    }

    // Store output
    if ((segment + threadIdx.x + BLOCK_DIM) < N)
    {
        output[segment + threadIdx.x] = buffer_s[threadIdx.x];
        output[segment + threadIdx.x + BLOCK_DIM] = buffer_s[threadIdx.x + BLOCK_DIM];
    }
    else if ((segment + threadIdx.x) < N)
    {
        output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    }
}

__global__ void add_kernel(float *output, float *partialSums, unsigned int N)
{
    unsigned int segment = 2 * blockIdx.x * blockDim.x;
    if (blockIdx.x > 0)
    {
        if ((segment + threadIdx.x + BLOCK_DIM) < N)
        {
            output[segment + threadIdx.x] += partialSums[blockIdx.x];
            output[segment + threadIdx.x + BLOCK_DIM] += partialSums[blockIdx.x];
        }
        else if ((segment + threadIdx.x) < N)
        {
            output[segment + threadIdx.x] += partialSums[blockIdx.x];
        }
    }
}

void scan_gpu_d(float *input_d, float *output_d, unsigned int N)
{

    Timer timer;

    // Configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;

    // Allocate partial sums
    startTime(&timer);
    float *partialSums_d;
    cudaMalloc((void **)&partialSums_d, numBlocks * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Partial sums allocation time");

    // Call kernel
    startTime(&timer);
    scan_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Scan partial sums then add
    if (numBlocks > 1)
    {

        // Scan partial sums
        scan_gpu_d(partialSums_d, partialSums_d, numBlocks);

        // Add scanned sums
        add_kernel<<<numBlocks, numThreadsPerBlock>>>(output_d, partialSums_d, N);
    }

    // Free memory
    startTime(&timer);
    cudaFree(partialSums_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}
