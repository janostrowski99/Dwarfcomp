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
  int ixy = blockIdx.x*blockDim.x+threadIdx.x;

  //printf("gpu %d and %d \n", ixy, genome[ixy]);

  int mutpose= (int) (curand_uniform(&state[ixy+2])*(lengh));


  int muttype= (int) (curand_uniform(&state[ixy+2])*(3))+1;


  //float test=curand_uniform(&state[ixy]+ixy)*(lengh);

  genome[mutpose+(blockIdx.x*lengh)]=(genome[mutpose+(blockIdx.x*lengh)]+muttype)%4;

  //printf("gpu %d\n", mutpose);
  //printf("gpu %f\n", test);
}

__global__ void grand(curandState *state,unsigned long seed)
{
  int ixy = blockIdx.x*blockDim.x+threadIdx.x;

  curand_init(seed,(ixy),0,&state[ixy+2]);
}

__global__ void alcate(int *genome,int lengh)
{
  int ixy = (blockIdx.x+1)*(blockDim.x)+threadIdx.x;

  genome[ixy]=genome[threadIdx.x];


  //printf("a %d %d\n",blockIdx.x,  genome[2]);

}
__global__ void alcateBig(int *genome,int lengh)
{
  int ixy = (blockIdx.x+1)*lengh;
  int ixy2 = threadIdx.x;
  int ixy3;
  int big=lengh/1000;
  for(int b=0;b<big;b++)
  {
    ixy3=ixy2+b*1000;

    if(ixy3<lengh)
    {
      //printf("ixy %d ixy3 %d b %d gen %d \n",ixy, ixy3,b,genome[ixy3]);
      genome[ixy3+ixy]=genome[ixy3];
    }


  }




  //printf("a %d %d\n",blockIdx.x,  genome[2]);

}


__global__ void calculateEntropy(int *genome,int posi, int types, int len, int *entropy)
{

  extern  __shared__ int temp[];

  //__shared__ int temp[12];
  int ixy = blockIdx.x*blockDim.x+threadIdx.x;
  int total=0;
  int multi;
  //printf("total %d", blockIdx.x);
  for(int j=ixy;j<ixy+len;j++)
  {
    multi=1;
    for(int i=0;i<j-ixy;i++)
    {
      multi=multi*types;
    }
    //printf("multi %d", types);
    total=total+(genome[j]*multi); //entropy state numbers
    //temp[total+(blockIdx.x*posi)]+=1; //MMMMMMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEHHHHHHHHHH

  }
  //printf("%d \n",blockDim.x);
  __threadfence();
  atomicAdd(temp+(total+(blockIdx.x*posi)),1);
  //temp[total+(blockIdx.x*posi)]+=1; //MMMMMMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEHHHHHHHHHH

  //temp[1]++;
  entropy[total+(blockIdx.x*posi)]=temp[total+(blockIdx.x*posi)];
  //printf("total %d %d %d \n",genome[ixy], entropy[total+(blockIdx.x*posi)], total+(blockIdx.x*posi));
  //printf("total %d %d %d \n",genome[ixy], entropy[total+(blockIdx.x*posi)],  temp[1]);
  __syncthreads();
}

__global__ void calculateEntropyBig(int *genome,int posi, int types, int len, int lengh, int *entropy)
{

  extern  __shared__ int temp[];

  //__shared__ int temp[12];
  int ixy = blockIdx.x*lengh;
  int ixy2 = threadIdx.x;
  int ixy3;
  int total=0;
  int big=lengh/1000;
  //printf("big%d", big);
  int multi;


  for(int b=0;b<big;b++)
  {
    ixy3=ixy2+b*1000;

    if(ixy3<lengh)
    {
      total=0;


    for(int j=ixy+ixy3;j<ixy+ixy3+len;j++)
    {
      multi=1;
      for(int i=0;i<j-ixy-ixy3;i++)
      {
        multi=multi*types;
      }
      //printf("multi %d", types);
      total=total+(genome[j]*multi); //entropy state numbers

      //temp[total+(blockIdx.x*posi)]+=1; //MMMMMMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEHHHHHHHHHH

    }
    //printf("ixy %d ixy3 %d b %d total %d \n",ixy, ixy3,b,total);
    atomicAdd(temp+(total+(blockIdx.x*posi)),1);
    entropy[total+(blockIdx.x*posi)]=temp[total+(blockIdx.x*posi)];
  }
  }
  __syncthreads();
}

__device__ double calculateEntropy2( int posi ,int* entab,int set) // possibilities and array for entropy
{
  double sum=0;
  int mv=set*posi;
  for(int i =0;i<posi;i++)
  {
    sum=sum+entab[i+mv];
  }

  double entropy=0;
  for(int i =0;i<posi;i++)
  {
    //printf("ent %d %d \n",set , entab[i+mv]);

    if((double)entab[i+mv]/sum>0)
    {
      entropy=entropy+(((double)entab[i+mv]/sum)*(double)log2((double)entab[i+mv]/sum));
    }

  }
  return -entropy;
}

__global__ void calculateEntropy3(int posi, int *entropy, double*out)
{
  int ixy = blockIdx.x*blockDim.x+threadIdx.x;

  //printf("%d %d\n",entropy[ixy],ixy);
  out[ixy]=calculateEntropy2(posi,entropy,ixy);
  //printf("entr %d %lf \n", ixy ,out[ixy]);

}




int main(int argc, char* argv[])
{
  int mutationnum;
  int lengh;
  int day;
  int *genome;
  cudaError_t err;


  //chengables
  int createmut=5; //how many mutations to create
  int nchain=4; //how many genoms to mutate
  int len=1; //leng of chcked chain
  int types=4; //number of type variables
  //~chengables

  sscanf(argv[1], "%d", &createmut);
  sscanf(argv[2], "%d", &nchain);
  sscanf(argv[3], "%d", &len);


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
  //lengh=150;

  genome=(int*)malloc(lengh*sizeof(int));


  for(int i=0;i<lengh;i++)
  {
    fscanf (ifile, "%d", &genome[i]);
    //printf("%d %d\n",i, genome[i]);
  }



  //cudaSetDevice(0);




  double *outentropy;
  int *entropytab;
  double enbuffer[posi][nchain];
  for(int i=0;i<posi;i++)
  {
    for(int j=0;j<nchain;j++)
    {
      enbuffer[i][j]=0;
    }
  }
  cudaMalloc(&entropytab, nchain*posi*sizeof(int));
  cudaMalloc(&outentropy, nchain*sizeof(double));
  cudaMemcpy(enbuffer,entropytab,nchain*posi*sizeof(int),cudaMemcpyHostToDevice);

  int *gpm;
  cudaMalloc(&gpm,nchain*lengh*sizeof(int));
  err=cudaMemcpy(gpm,genome,lengh*sizeof(int),cudaMemcpyHostToDevice);

  if(err == cudaErrorInvalidValue)
  printf("1!\n");
  else if(err == cudaErrorInvalidDevicePointer)
  printf("2!\n");
  else if(err == cudaErrorInvalidMemcpyDirection)
  printf("3!\n");

  cudaMalloc((void **)&state,(createmut+2)*sizeof(curandState));
  cudaDeviceSynchronize();

  if(lengh<1024)
  {
    alcate<<<  nchain-1,lengh >>>(gpm,lengh);
  }
  else
  {
    alcateBig<<<  nchain-1,1000 >>>(gpm,lengh);
  }


  cudaDeviceSynchronize();
  grand<<< createmut, nchain >>>(state,unsigned(time(NULL)));
  cudaDeviceSynchronize();
  mutate<<< createmut, nchain >>>(gpm,lengh,state);
  cudaDeviceSynchronize();


  if(lengh<1024)
  {
    calculateEntropy<<<  nchain, lengh+1-len,posi*nchain*sizeof(int) >>>(gpm,posi,types,len,entropytab);
  }
  else
  {
    calculateEntropyBig<<<  nchain, 1000 ,posi*nchain*sizeof(int) >>>(gpm,posi,types,len,lengh,entropytab);
  }

  //calculateEntropy<<<  nchain, lengh+1-len,128 >>>(gpm,posi,types,len,entropytab);
  //calculateEntropy<<<  nchain, lengh+1-len >>>(gpm,posi,types,len,entropytab,temp);
  cudaDeviceSynchronize();
  calculateEntropy3<<< 1, nchain >>>(posi,entropytab,outentropy);
  cudaDeviceSynchronize();

  //cudaMemcpy(genome,gpm,lengh*sizeof(int),cudaMemcpyDeviceToHost);
  //cudaMemcpy(enbuffer,entropy,posi*sizeof(double),cudaMemcpyDeviceToHost);

  double outentropy2[nchain];

  cudaMemcpy(outentropy2,outentropy,nchain*sizeof(double),cudaMemcpyDeviceToHost);
  for(int i=0;i<nchain;i++)
  {
    printf("entr %lf\n", outentropy2[i]);
  }





  cudaDeviceSynchronize();

  return 1;
}
