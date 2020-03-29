#include <stdio.h>

// 1. Note the convention d_* is used for device and h_* is used for host allocations.
// 2. __global__ tells cuda that what follows is a kernel implementation


// Cuda kernel that returns a cube of a given array
// Mostly written in a serial manner
__global__ void cube(float *d_out, float *d_in) {
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f * f * f;
}

int main(int argc, char **argv) {
  const int ARRAY_SIZE = 1000;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  // Create input array on the host.
  float h_in[ARRAY_SIZE];
  for (int i=0; i < ARRAY_SIZE; i++) {
    h_in[i] = i;
  }
  float h_out[ARRAY_SIZE];

  // Declare pointers for GPU memory
  float *d_in;
  float *d_out;

  // Allocate memory on GPU.
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_out, ARRAY_BYTES);

  // Transfer array to GPU.
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  // Launch the kernel
  cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

  // Copy back the result
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  // Print the resulting array
  for(int i=0; i < ARRAY_SIZE; i++) {
     printf("%f", h_out[i]);
     printf(((i % 4) != 3) ? "\t" : "\n");
  }

  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}

