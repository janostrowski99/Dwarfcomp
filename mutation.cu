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
#include <curand.h>
#include <curand_kernel.h>

__global__ void mutate(int *genome,int lengh,curandState *state)
{
  size_t ixy = blockIdx.x*blockDim.x+threadIdx.x;

  //printf("gpu %d and %d \n", ixy, genome[ixy]);

  int mutpose= (int) (curand_uniform(&state[ixy+2])*(lengh));


  int muttype= (int) (curand_uniform(&state[ixy+2])*(3))+1;

  //float test=curand_uniform(&state[ixy]+ixy)*(lengh);

  genome[mutpose]=(genome[mutpose]+muttype)%4;

  printf("gpu %d\n", mutpose);
  //printf("gpu %f\n", test);
}

__global__ void grand(curandState *state,unsigned long seed)
{
    size_t ixy = blockIdx.x*blockDim.x+threadIdx.x;

    curand_init(seed,(ixy),0,&state[ixy+2]);
}


int main()
{
  int mutationnum;
  int lengh;
  int day;
  int *genome;

  int createmut=5;
  curandState *state;




  FILE* ifile = fopen ("genome.txt", "r");

  fscanf (ifile, "%d", &mutationnum);
  fscanf (ifile, "%d", &day);
  fscanf (ifile, "%d", &lengh);


  genome=(int*)malloc(lengh*sizeof(int));


    for(int i=0;i<lengh;i++)
    {
      fscanf (ifile, "%d", &genome[i]);
    }

    cudaSetDevice(0);


    int *gpm;
    cudaMalloc(&gpm,  lengh*sizeof(int));

    cudaMalloc((void **)&state,(createmut+2)*sizeof(curandState));
    cudaMemcpy(gpm,genome,lengh*sizeof(int),cudaMemcpyHostToDevice);
    grand<<< createmut, 1 >>>(state,unsigned(time(NULL)));
    cudaDeviceSynchronize();
    mutate<<< createmut, 1 >>>(gpm,lengh,state);
    cudaMemcpy(genome,gpm,lengh*sizeof(int),cudaMemcpyDeviceToHost);







    cudaDeviceSynchronize();

    return 1;
}
