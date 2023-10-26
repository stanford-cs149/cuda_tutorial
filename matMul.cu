#include "stdio.h"
#include "stdlib.h"

#define BLOCKDIM 32
#define BLOCKDIMP BLOCKDIM+1

//naive/slow kernel
__global__ void multiplyMatrices1Naive(float *a, float *b, float *c, int N){

	int xid = blockIdx.x*blockDim.x + threadIdx.x;
	int yid = blockIdx.y*blockDim.x + threadIdx.y;
  
	if (xid<N && yid<N){
		float result = 0;
		for (int i=0; i<N; i++){
			result += a[xid*N+i]*b[i*N+yid];
		}
		c[xid*N+yid] = result;
	}
	return ;
}


//kernel will be faster as it uses shared memory
__global__ void multiplyMatrices2Shared(float *a, float *b, float *c, int N){

	int xid = blockIdx.x*blockDim.x + threadIdx.x;
	int yid = blockIdx.y*blockDim.y + threadIdx.y;
  	int xtid = threadIdx.x;
	int ytid = threadIdx.y;

	int nTiles = gridDim.x;
	__shared__ float aS[BLOCKDIM][BLOCKDIM];
	__shared__ float bS[BLOCKDIM][BLOCKDIM];

	if (xid>=N || yid>=N) return;

	float result = 0;
  	for (int k1=0; k1<nTiles; k1++)
	{
		int aRow = xid;
		int bCol = yid;

		int aCol = k1*blockDim.x + xtid;
		int bRow = k1*blockDim.x + ytid;

		//load tiles into shared memory
		aS[xtid][ytid] = a[ aRow*N + aCol ];
		bS[xtid][ytid] = b[ bRow*N + bCol ];
		__syncthreads();

		for (int k=0; k<blockDim.x; k++){
			result += aS[xtid][k]*bS[k][ytid];
		}
	}
	c[xid*N+yid] = result;
	
	return ;
}

//kernel will be faster as it uses shared memory and 
//pads shared memory to avoid bank conflict
__global__ void multiplyMatrices3Pad(float *a, float *b, float *c, int N){

	int xid = blockIdx.x*blockDim.x + threadIdx.x;
	int yid = blockIdx.y*blockDim.y + threadIdx.y;
  	int xtid = threadIdx.x;
	int ytid = threadIdx.y;

	int nTiles = gridDim.x;
	__shared__ float aS[BLOCKDIM][BLOCKDIMP];
	__shared__ float bS[BLOCKDIM][BLOCKDIMP];

	if (xid>=N || yid>=N) return;

	float result = 0;
  	for (int k1=0; k1<nTiles; k1++)
	{
		int aRow = xid;
		int bCol = yid;

		int aCol = k1*blockDim.x + xtid;
		int bRow = k1*blockDim.x + ytid;

		aS[xtid][ytid] = a[ aRow*N + aCol ];
		bS[xtid][ytid] = b[ bRow*N + bCol ];
		__syncthreads();

		for (int k=0; k<blockDim.x; k++){
			result += aS[xtid][k]*bS[k][ytid];
		}
	}
	c[xid*N+yid] = result;
	
	return ;
}

//kernel will be faster as it uses shared memory, avoids bank conflicts
//and allows for coalesced accesses in shared memory
__global__ void multiplyMatrices4Coalesce(float *a, float *b, float *c, int N){

	int xid = blockIdx.x*blockDim.x + threadIdx.x;
	int yid = blockIdx.y*blockDim.y + threadIdx.y;
  	int xtid = threadIdx.x;
	int ytid = threadIdx.y;

	int nTiles = gridDim.x;
	__shared__ float aS[BLOCKDIM][BLOCKDIMP];
	__shared__ float bS[BLOCKDIM][BLOCKDIMP];

	if (xid>=N || yid>=N) return;

	float result = 0;
  	for (int k1=0; k1<nTiles; k1++)
	{
		int aRow = xid;
		int bCol = yid;

		int aCol = k1*blockDim.x + xtid;
		int bRow = k1*blockDim.x + ytid;

		//transpose the b matrix into shared memory
		//this allows for better accessing of shared memory
		aS[xtid][ytid] = a[ aRow*N + aCol ];
		bS[ytid][xtid] = b[ bRow*N + bCol ];
		__syncthreads();

		for (int k=0; k<blockDim.x; k++){
			result += aS[xtid][k]*bS[ytid][k];
		}
	}
	c[xid*N+yid] = result;
	
	return ;
}





int main(){

	int N =1024; //size of each dimension
	int N2 = N*N; //size of total matrix

	float *a_cpu = (float*) malloc(sizeof(float)*N2);
	float *b_cpu = (float*) malloc(sizeof(float)*N2);
	float *c_cpu = (float*) malloc(sizeof(float)*N2);

	//let's fill em up with values, this can also be done on the gpu
	for (int i=0; i<N2; i++)
	{
		a_cpu[i] = 1.0f;
		b_cpu[i] = 1.0f;
	}


	//now we have to allocate memory on the gpu
	float *a;
	float *b;	
	float *c;

	cudaMalloc( &a, sizeof(float)*N2);
	cudaMalloc( &b, sizeof(float)*N2);
	cudaMalloc( &c, sizeof(float)*N2);

	//send the cpu memory to the gpu memory
	cudaMemcpy(a, a_cpu, N2*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b, b_cpu, N2*sizeof(float), cudaMemcpyHostToDevice);

	//always choose a multiple of 32 threads for blocksize
	int blockSize = 32;
	int nBlocks = (N + blockSize - 1 ) / blockSize;
	dim3 grid (nBlocks, nBlocks);
	dim3 block (blockSize, blockSize);

	printf("blocks %i, threads per block %i\n", nBlocks, blockSize);
	//launch your gpu kernel/function
	for (int i=0; i<50; i++){
		multiplyMatrices1Naive<<<grid, block>>>(a, b, c, N);
		multiplyMatrices2Shared<<<grid, block>>>(a, b, c, N);
		multiplyMatrices3Pad<<<grid, block>>>(a, b, c, N);
		multiplyMatrices4Coalesce<<<grid, block>>>(a, b, c, N);
	}
	//now let's bring the result vector back the the cpu
	cudaMemcpy(c_cpu, c, N2*sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("print out first 10 results, should all be 2\n");
	for (int i=0; i<10; i++){
		printf("result %i: %f\n", i, c_cpu[i]);
	}

	return 0;
}
