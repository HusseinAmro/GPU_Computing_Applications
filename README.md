# GPU_Computing Applications

## Overview

This repository contains a collection of GPU computing kernels developed as part of weekly individual assignments in CMPS224 GPU Computing course at the American University of Beirut (AUB). The primary objective was to learn CUDA programming and apply various optimization techniques to enhance performance.

## Assignemnts

1. **Element Wise Max of Two Vectors:** compute a vector c from two vectors a and b such that c[i] is the maximum of a[i] and b[i] for all i.
2. **Simple Matrix-Matrix Multiplication:** takes a matrix A of size M×K and a matrix B of size K×N, and produces a matrix C.
3. **Tiled Matrix-Matrix Multiplication:** uses shared memory tiling and expected to work for any set of matrix dimensions. 
4. **Tiled Convolution:** uses shared memory tiling and expected to work for any set of input dimensions.
5. **Histogram:** implement a histogram operation using atomic operations, and optimize it using privatization, shared memory, and thread coarsening. 
6. **Reduction:** with the following optimizations applied, memory coalescing, eliminating control divergence, using shared memory, and using warp shuffle instructions. 
7. **Exclusive Scan Brent-Kung Method:** Perform exclusive scan using the Brent-Kung method.

## Acknowledgement

- **main.cu:** contains setup and sequential code.
- **kernel.cu:** parallel implementation. 
- **common.h:** for shared declarations across main.cu and kernel.cu .
- **timer.h:** to assist with timing.
- **Makefile:** used for compilation.


## Instructions

- **To compile:**

  ```
  make
  ```

- **To run 1:**

  ```
  ./vecmax
  ```
  - For testing on different input sizes, you can provide your own value for the number of 
  vector elements: ./vecmax <.M> (example: ./vecmax 1000000)


- **To run 2:**

  ```
  ./mm
  ```
  - For testing on different matrix sizes, you can provide your own values for matrix 
  dimensions as follows: ./mm <.M> <.N> <.K> (example: ./mm 256 512 128)


- **To run 3:**

  ```
  ./mm-tiled
  ```
  For testing on different matrix sizes, you can provide your own values for matrix 
  dimensions as follows: ./mm-tiled <.M> <.N> <.K> 


- **To run 4:**

  ```
  ./convolution
  ```
  For testing on different input sizes, you can provide your own values for the input 
  dimensions as follows: ./convolution <.height> <.width>


- **To run 5:**

  ```
  ./histogram
  ```
  For testing on different input sizes, you can provide your own values for the input 
  dimensions as follows: ./histogram <.height> <.width>


- **To run 6:**

  ```
  ./reduction
  ```
  For testing on different input sizes, you can provide your own values for input size as 
  follows: ./reduction <.N>


- **To run 7:**

  ```
  ./scan
  ```
  For testing on different input sizes, you can provide your own values for input size as 
  follows: ./scan <.N>