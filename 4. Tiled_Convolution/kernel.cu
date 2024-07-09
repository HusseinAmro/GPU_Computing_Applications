
#include "common.h"

#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - (2 * FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float *input, float *output, unsigned int width, unsigned int height)
{
    int row = blockIdx.y * OUT_TILE_DIM - FILTER_RADIUS + threadIdx.y;
    int col = blockIdx.x * OUT_TILE_DIM - FILTER_RADIUS + threadIdx.x;
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row < height && row >= 0 && col < width && col >= 0)
    {
        in_s[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
    else
    {
        in_s[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    if (row < height && row >= 0 && col < width && col >= 0)
    {
        if (threadIdx.x >= FILTER_RADIUS && threadIdx.x < IN_TILE_DIM - FILTER_RADIUS && threadIdx.y >= FILTER_RADIUS && threadIdx.y < IN_TILE_DIM - FILTER_RADIUS)
        {
            float sum = 0.0f;
            for (unsigned int filterRow = 0; filterRow < FILTER_DIM; ++filterRow)
            {
                for (unsigned int filterCol = 0; filterCol < FILTER_DIM; ++filterCol)
                {
                    int inRow = threadIdx.y - FILTER_RADIUS + filterRow;
                    int inCol = threadIdx.x - FILTER_RADIUS + filterCol;
                    sum += filter_c[filterRow][filterCol] * in_s[inRow][inCol];
                }
            }
            output[row * width + col] = sum;
        }
    }
}

void copyFilterToGPU(float filter[][FILTER_DIM])
{
    // Copy filter to constant memory
    cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM * FILTER_DIM * sizeof(float));
}

void convolution_tiled_gpu(float *input_d, float *output_d, unsigned int width, unsigned int height)
{
    // Call kernel
    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_tiled_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
}
