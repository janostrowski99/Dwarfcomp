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


__global__ void calculateEntropy(int *genome,int posi, int types, int len, double *entropy)
{
    size_t ixy = blockIdx.x*blockDim.x+threadIdx.x;

    int total=0;
    int multi;
    for(int j=ixy;j<ixy+len;j++)
    {
      multi=1;
      for(int i=0;i<j-ixy;i++)
      {
        multi=multi*types;
      }
      total=total+(genome[j]*multi); //entropy state numbers

    }
    entropy[total]++;
}

double calculateEntropy( int posi ,int* entab) // possibilities and array for entropy
{
  double sum=0;
  for(int i =0;i<posi;i++)
  {
    sum=sum+entab[i];
  }
  double entropy=0;
  for(int i =0;i<posi;i++)
  {
    //cout<<entropy;
    if((double)entab[i]/sum)
    {
      entropy=entropy+(((double)entab[i]/sum)*(double)TMath::Log2((double)entab[i]/sum));
    }

  }
  return -entropy;
}


int main()
{
  int mutationnum;
  int lengh;
  int day;
  int *genome;


  //chengables
  int createmut=5; //how many mutations to create
  int len=1; //leng of chcked chain
  int types=4; //number of type variables
  //~chengables

  int posi=types;
  for(int i=1;i<len;i++)
  {
    posi=posi*types;
  }


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
    cudaMemcpy(gpm,genome,lengh*sizeof(int),cudaMemcpyHostToDevice);

    double *entropy;
    double enbuffer[posi];
    for(int i=0;i<posi;i++)
    {
      enbuffer[i]=0;
    }
    cudaMalloc(&entropy,  posi*sizeof(double));
    cudaMemcpy(entropy,enbuffer,posi*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void **)&state,(createmut+2)*sizeof(curandState));

    grand<<< createmut, 1 >>>(state,unsigned(time(NULL)));
    cudaDeviceSynchronize();
    mutate<<< createmut, 1 >>>(gpm,lengh,state);
    cudaDeviceSynchronize();
    calculateEntropy<<< lengh+1-len, 1 >>>(gpm,posi,types,len,entropy);
    cudaDeviceSynchronize();

    cudaMemcpy(genome,gpm,lengh*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(enbuffer,entropy,posi*sizeof(double),cudaMemcpyDeviceToHost);






    cudaDeviceSynchronize();

    return 1;
}
