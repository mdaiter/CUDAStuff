#include <stdio.h>
#include <cuda.h>
#include<math.h>

// Kernel that executes on the CUDA device
__global__ void square_array(float *a, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) a[idx] = a[idx] * a[idx];
}

__global__ void reduce_partials(float *d_in, int step)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx % (2 * step) == 0){
    int stepForward = pow ((float)2, (float)step);
    d_in[idx] = d_in[idx] + d_in[idx + stepForward];
  }
}

void reduce_array(float* d_in){
  size_t sizeOfArr = sizeof(d_in) / sizeof(float);
  int steps = floor(log(sizeOfArr) / log(2));

  int numThreads = 0;
  if (sizeOfArr < 1024) {
    numThreads = floor(sizeOfArr / 2);
  } else {
    numThreads = 512;
  }

  int numBlocks = floor(sizeOfArr / 1024);
  for (int i = 0 ; i < steps; i++){
    reduce_partials<<< numBlocks, numThreads>>>(d_in, i);
  }
}

// main routine that executes on the host
int main(void)
{
  float *a_h, *a_d;  // Pointer to host & device arrays
  const int N = 16;  // Number of elements in arrays
  size_t size = N * sizeof(float);
  a_h = (float *)malloc(size);        // Allocate array on host
  cudaMalloc((void **) &a_d, size);   // Allocate array on device
  // Initialize host array and copy it to CUDA device
  for (int i=0; i<N; i++) a_h[i] = (float)i;
  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  // Do calculation on device:
  reduce_array(a_d);
  // Retrieve result from device and store it in host array
  cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
  // Print results
  for (int i=0; i<N; i++) printf("%d %f\n", i, a_h[i]);
  // Cleanup
  free(a_h); cudaFree(a_d);
}
