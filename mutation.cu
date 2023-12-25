/**

  nvcc -arch sm_50 -O3 mutation.cu -o mutation -lcuda -lcufft
 *
 * */

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void mutate(int *genome)
{
  size_t ixy = blockIdx.x*blockDim.x+threadIdx.x;

  //printf("gpu %d and %d \n", ixy, genome[ixy]);
  printf("gpu %d\n", genome[ixy]);
  genome[ixy]++;

}

int main()
{

    int lengh=3;
    int *genome;
    genome=(int*)malloc(lengh*sizeof(int));

    for(int i=0;i<lengh;i++)
    {
      genome[i]=i+1;
    }


    printf("test %d\n", genome[2]);

    cudaSetDevice(0);


    int *gpm;
    cudaMalloc(&gpm,  lengh*sizeof(int));


    cudaMemcpy(gpm,genome,lengh*sizeof(int),cudaMemcpyHostToDevice);
    mutate<<< lengh, 1 >>>(gpm);
    cudaMemcpy(genome,gpm,lengh*sizeof(int),cudaMemcpyDeviceToHost);



    printf("test %d\n", genome[2]);



    cudaDeviceSynchronize();

    return 1;
}
