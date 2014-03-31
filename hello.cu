#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>

__global__ void hello(int* d_in){
  d_in[threadIdx.x] = (blockIdx.x + threadIdx.x) * (blockIdx.x * threadIdx.x);
  printf("%d : %d\n", threadIdx.x, d_in[threadIdx.x]);
}

int main(){
    int* h_array = (int*)malloc(1200*1024 * sizeof(int));
    int* d_array;
    cudaMalloc((void**) &d_array, 1200*1024 * sizeof(int));
    /*for (int i = 0; i < 300; i++){
      h_array[i] = i;
    }*/
    cudaMemcpy(d_array, h_array, 1200*1024*sizeof(int), cudaMemcpyHostToDevice);

    hello<<<1200,1024>>>(d_array);
    cudaMemcpy(h_array, d_array, sizeof(int) * 1200*1024, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    free(h_array);
    cudaFree(d_array);
    return 0;
}
