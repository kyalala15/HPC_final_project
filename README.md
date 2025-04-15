# HPC Final Project: Sobel Edge Detection Optimization

This project implements and optimizes the Sobel edge detection algorithm using OpenMP for parallel processing. The implementation includes both a baseline version and an optimized version with various performance improvements.

## Project Structure

### `sobel`
Contains the baseline implementation of the Sobel edge detection algorithm:
- `sobel_edge_detection.cpp`: Standard implementation using 2D vectors
- `Makefile`: Compilation instructions for the baseline version

### `sobel_optimized`
Contains the optimized implementation with performance improvements:
- `sobel_optimized.cpp`: Optimized implementation using 1D vectors and cache-friendly access patterns
- `Makefile`: Compilation instructions for the optimized version

### `images`
Contains test images in PGM format for processing:
- `ball.pgm`: Sample image for testing
- `pat1000.pgm`: 1000x1000 pixel test image
- `pat2000.pgm`: 2000x2000 pixel test image
- `pat3000.pgm`: 3000x3000 pixel test image
- `pat4000.pgm`: 4000x4000 pixel test image
- `pat8000.pgm`: 8000x8000 pixel test image

## Optimization Techniques

The optimized version includes the following improvements:

1. **Memory Layout**: Changed from 2D vectors to 1D vectors for better cache locality
2. **Loop Unrolling**: Explicit unrolling of the convolution loops
3. **Index Pre-computation**: Pre-computing row indices to reduce multiplication operations
4. **Data Locality**: Improved data access patterns for better cache utilization

## Running on Explorer

To allocate a node on Explorer, use the following command:
`srun --cpus-per-task=28 --cpu-bind=cores --partition=courses --pty --time=01:00:00 --nodes=1 /bin/bash`

After allocating a node, you can compile and run the code as follows:

1. **Compile the baseline version**:
   cd sobel
   make run

2. **Compile the optimized version**:
   cd ../sobel_optimized
   make run

## Modifying Input Parameters

To change the input image or number of threads:

1. Edit the `#define INPUT_FILE_PATH` in the source files to point to a different image
2. Edit the `#define NUM_THREADS` in the source files 

## References

1. [Sobel Edge Detection](https://medium.com/@twinnroshan/understanding-and-implementing-edge-detection-in-c-with-sobel-operator-31159f26587c)
