#include "stdio.h"
#include "stdlib.h"

__global__ void addVectors(float *a, float *b, float *c, int N){

	//get out global thread id 
	//threads are part of blocks, blocks are equally sized
	//groups of threads. blocks are part of our 'grid'
	//each block has an ID, and each thread has an ID
	//to get our global thread, multiply block ID with blocksize
	//then add in your thread id

	int pid = blockIdx.x*blockDim.x + threadIdx.x;

	//we may have some threads left over because we always 'spawn'
	//threads a block at a time, and blocks have constant size
	//therefore if blocksize=32, we spawn a multiple of 32 number of threads 
       
	//so we'll have up to 31 threads that do nothing, so we put in this if statement
	//so they do nothing, otherwise they may try to access memory out of bounds
	if (pid<N){

		c[pid] = a[pid] + b[pid];
	}
	return ;
}


int main(){

	int N = 100000; //one hundred thousand

	//data from the gpu must generally come from the cpu first
	//so we allocate arrays on the cpu, and then
	//use special functions/memcopies to transfer them
	float *a_cpu = (float*) malloc(sizeof(float)*N);
	float *b_cpu = (float*) malloc(sizeof(float)*N);
	float *c_cpu = (float*) malloc(sizeof(float)*N);

	//let's fill em up with values, this can also be done on the gpu
	for (int i=0; i<N; i++)
	{
		a_cpu[i] = 1.0f;
		b_cpu[i] = 1.0f;
	}


	//now we have to allocate memory on the gpu
	float *a;
	float *b;
	float *c;
	cudaMalloc( &a, sizeof(float)*N);
	cudaMalloc( &b, sizeof(float)*N);
	cudaMalloc( &c, sizeof(float)*N);

	//send the cpu memory to the gpu memory
	cudaMemcpy(a, a_cpu, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b, b_cpu, N*sizeof(float), cudaMemcpyHostToDevice);

	//always choose a multiple of 32 threads for blocksize
	int blockSize = 64;
	int nBlocks = (N + blockSize - 1 ) / blockSize;

	printf("blocks %i, threads per block %i\n", nBlocks, blockSize);
	//launch your gpu kernel/function
	addVectors<<<nBlocks,blockSize>>>(a, b, c, N);

	//now let's bring the result vector back the the cpu
	cudaMemcpy(c_cpu, c, N*sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("print out first 10 results, should all be 2\n");
	for (int i=0; i<10; i++){
		printf("result %i: %f\n", i, c_cpu[i]);
	}

	return 0;
}
