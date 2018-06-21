/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs,
 * with an emphasis on simple illustration of the techniques (not on performance).
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the
 * application. On the other side, you can still extend your desktop to screens
 * attached to both GPUs.
 */

// System includes
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <mpi.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <timer.h>
#include "cudafunc.h"
#include "dataprocess.h"

using namespace std;

typedef struct
{
    float *fai_h;
    float *send_u, *send_d;
    float *fai_d, *fai_d_n;
    int flag; 
    int hoa;     
    float *temp;
    cudaStream_t stream;
} TGPUG;


////////////////////////////////////////////////////////////////////////////////
// GPU calculation kernel
////////////////////////////////////////////////////////////////////////////////

__device__ float SingleFai(float *fai, unsigned int i,unsigned int j,size_t pitch)
{
	float *a = (float*)((char*)fai + (i - 1)*pitch);
	float *b = (float*)((char*)fai + (i + 1)*pitch);
	float *c = (float*)((char*)fai + i*pitch);
	return ((a[j] + b[j] + c[j - 1] + c[j + 1]) / 4);
}


__global__ void SingleNodeFaiIter(float *fai, float *fai_n, size_t pitch, int H, int W, int flag) 
{
	//unsigned int i = blockDim.y*blockIdx.y + threadIdx.y;
	//unsigned int j = blockDim.x*blockIdx.x + threadIdx.x;
	for (int i = blockDim.y*blockIdx.y + threadIdx.y; i < H; i += blockDim.y*gridDim.y) {
		float *fai_row_n = (float*)((char*)fai_n + i*pitch);
		for (int j = blockDim.x*blockIdx.x + threadIdx.x; j < W; j += blockDim.x*gridDim.x) {
			if(flag==0){
				if (i > 1 && i < H - 1 && j > 0 && j < W - 1)
				    fai_row_n[j] = SingleFai(fai, i, j, pitch);
			}
			else if(flag==1){
				if (i > 0 && i < H - 2 && j > 0 && j < W - 1)
				    fai_row_n[j] = SingleFai(fai, i, j, pitch);
			}
            else{
                if (i > 0 && i < H - 1 && j > 0 && j < W - 1)
                    fai_row_n[j] = SingleFai(fai, i, j, pitch);
            }
		}
	}
}   


////////////////////////////////////////////////////////////////////////////////
// Device information
////////////////////////////////////////////////////////////////////////////////
void GetDeviceName(int count) 
{ 
    cudaDeviceProp prop;
    for(int i= 0;i< count;++i)
    {
        cudaGetDeviceProperties(&prop,i) ;
        cout << "Device "<<i<<" name is :" << prop.name<< endl;
    } 
}


        

////////////////////////////////////////////////////////////////////////////////
// Calculate on GPU
////////////////////////////////////////////////////////////////////////////////

void GpuCalculate(float *fai, int H, int W, int my_rank, int comm_sz)
{

    TGPUG G[MAX_GPU_COUNT];
    int i, n;
    int GPU_N;
    size_t pitch;
    int DH, DW, DWP;
    //send offset address, recieve offset address
    int soa_u, soa_d, roa_u, roa_d;  
    //Receive buffer
    float *recv_u, *recv_d;

    MPI_Status status;

    const dim3 blockDim(32, 16,1);
	const dim3 gridDim(8, 8,1);

    checkCudaErrors(cudaGetDeviceCount(&GPU_N));

    if (GPU_N > MAX_GPU_COUNT){
        GPU_N = MAX_GPU_COUNT;
    }

    //Get data sizes for each GPU
    DH = H / GPU_N + 2;
    DW = W;

    //Allocate receive memory on host
    checkCudaErrors(cudaMallocHost((void **)&recv_u, DW * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&recv_d, DW * sizeof(float)));

    memset(recv_u, 0, DW*sizeof(float));
    memset(recv_d, 0, DW*sizeof(float));

    //Subdividing total data across GPUs, Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked) 
    for (i = 0; i < GPU_N; i++){

	    G[i].fai_h = fai + i * (H / GPU_N) * W;
        G[i].flag = 2;
	    //set device
        checkCudaErrors(cudaSetDevice(i));
        //creat stream
        checkCudaErrors(cudaStreamCreate(&G[i].stream));
        //Allocate device memory
        checkCudaErrors(cudaMallocPitch((void **)&G[i].fai_d, &pitch, DW * sizeof(float), DH));
        checkCudaErrors(cudaMallocPitch((void **)&G[i].fai_d_n, &pitch, DW * sizeof(float), DH));
        //Allocate send memory on host
        checkCudaErrors(cudaMallocHost((void **)&G[i].send_u, DW * sizeof(float)));
        checkCudaErrors(cudaMallocHost((void **)&G[i].send_d, DW * sizeof(float)));

        if(i==0){
        	G[i].hoa = W;
            if(my_rank==0)
                G[i].flag = 0;
        }
        else{
            G[i].hoa = (H/2 + 1)*W;
            if(my_rank==comm_sz-1)
                G[i].flag = 1;
        }       
    }

    // Address offset
    DWP = pitch/sizeof(float);
    soa_u = DWP;
    soa_d = (DH-2)*DWP;
    roa_u = 0;
    roa_d = (DH-1)*DWP;

    //Start compute on GPU(s)
    if(my_rank==0)
        cout<<"Computing with "<<comm_sz * GPU_N<<" GPUs on "<<comm_sz<<" Nodes..."<<endl;    

    //Copy initial data to GPU
    for (i = 0; i < GPU_N; i++){
    	//Set device
        checkCudaErrors(cudaSetDevice(i));
        //Copy initial data from CPU
		checkCudaErrors(cudaMemcpy2D(G[i].fai_d, pitch, G[i].fai_h, DW * sizeof(float), DW * sizeof(float), DH, cudaMemcpyHostToDevice));		
		checkCudaErrors(cudaMemcpy2D(G[i].fai_d_n, pitch, G[i].fai_h, DW * sizeof(float), DW * sizeof(float), DH, cudaMemcpyHostToDevice));
    }

    //Launch the kernel and copy boundary data back. All asynchronously
    for (n = 0; n < 5000; n++){
    	for (i = 0; i < GPU_N; i++){
    		//Set device
    		checkCudaErrors(cudaSetDevice(i));
    		//Perform GPU computations
        	SingleNodeFaiIter<<<gridDim, blockDim, 0, G[i].stream>>>(G[i].fai_d, G[i].fai_d_n, pitch, DH, DW, G[i].flag);
        	//Read back boundary data from GPU
        	checkCudaErrors(cudaMemcpy2DAsync(G[i].send_u, DW * sizeof(float), G[i].fai_d_n + soa_u, pitch, 
        									  DW * sizeof(float), 1, cudaMemcpyDeviceToHost, G[i].stream));
            checkCudaErrors(cudaMemcpy2DAsync(G[i].send_d, DW * sizeof(float), G[i].fai_d_n + soa_d, pitch, 
                                              DW * sizeof(float), 1, cudaMemcpyDeviceToHost, G[i].stream));
        }        
        
        for (i = 0; i < GPU_N; i++){
        	//Set device
        	checkCudaErrors(cudaSetDevice(i));
		    //Wait for all operations to finish
        	cudaStreamSynchronize(G[i].stream);
   
        	G[i].temp = G[i].fai_d;
        	G[i].fai_d = G[i].fai_d_n;
        	G[i].fai_d_n = G[i].temp;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(my_rank==0){
            MPI_Sendrecv(G[i].send_d, DW, MPI_FLOAT, 1, 0, 
                         recv_d, DW, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status);
 //          MPI_Send(G[1].send_d, DW, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
 //          MPI_Recv(recv_d, DW, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status);
        }

        else if(my_rank==comm_sz-1){
            MPI_Sendrecv(G[i].send_u, DW, MPI_FLOAT, my_rank-1, 1, 
                         recv_u, DW, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, &status);
 //           MPI_Send(G[0].send_u, DW, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD);
 //           MPI_Recv(recv_u, DW, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, &status);
        }
        else{
            MPI_Sendrecv(G[0].send_u, DW, MPI_FLOAT, my_rank-1, 1, 
                         recv_d, DW, MPI_FLOAT, my_rank+1, 1, MPI_COMM_WORLD, &status);
            MPI_Sendrecv(G[0].send_d, DW, MPI_FLOAT, my_rank+1, 0, 
                         recv_u, DW, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, &status);
//            MPI_Send(G[0].send_u, DW, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD);
//            MPI_Recv(recv_d, DW, MPI_FLOAT, my_rank+1, 1, MPI_COMM_WORLD, &status);
//            MPI_Send(G[1].send_d, DW, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD);
//            MPI_Recv(recv_u, DW, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, &status);
        }

        //Write new boundary value to GPU
        for (i = 0; i < GPU_N; i++){
            //Set device
        	checkCudaErrors(cudaSetDevice(i));

            int j=(i==0)?1:0;

        	if (i == 0){
			    checkCudaErrors(cudaMemcpy2DAsync(G[i].fai_d + roa_d, pitch, G[j].send_u, DW * sizeof(float), 
											      DW * sizeof(float), 1, cudaMemcpyHostToDevice, G[i].stream)); 
                checkCudaErrors(cudaMemcpy2DAsync(G[i].fai_d + roa_u, pitch, recv_u, DW * sizeof(float), 
                                                  DW * sizeof(float), 1, cudaMemcpyHostToDevice, G[i].stream));                
            }
            else{
                checkCudaErrors(cudaMemcpy2DAsync(G[i].fai_d + roa_d, pitch, recv_d, DW * sizeof(float), 
                                                  DW * sizeof(float), 1, cudaMemcpyHostToDevice, G[i].stream)); 
                checkCudaErrors(cudaMemcpy2DAsync(G[i].fai_d + roa_u, pitch, G[j].send_d, DW * sizeof(float), 
                                                  DW * sizeof(float), 1, cudaMemcpyHostToDevice, G[i].stream));                
            }
        }        
    }


    for (i = 0; i < GPU_N; i++){
    	//Set device
    	checkCudaErrors(cudaSetDevice(i));
        //Wait for all operations to finish
        cudaStreamSynchronize(G[i].stream);
    	//Read back final data from GPU
    	checkCudaErrors(cudaMemcpy2D(fai + G[i].hoa, W * sizeof(float), G[i].fai_d + DWP , pitch, 
									 DW * sizeof(float), DH-2, cudaMemcpyDeviceToHost)); 
    }

    //Process GPU results
    for (i = 0; i < GPU_N; i++){
        //Set device
        checkCudaErrors(cudaSetDevice(i));
        //Shut down this GPU
        checkCudaErrors(cudaFreeHost(G[i].send_u));
        checkCudaErrors(cudaFreeHost(G[i].send_d));
        checkCudaErrors(cudaFree(G[i].fai_d));
        checkCudaErrors(cudaFree(G[i].fai_d_n));
        checkCudaErrors(cudaStreamDestroy(G[i].stream));
    }

    // clean up
    checkCudaErrors(cudaFreeHost(recv_u));
    checkCudaErrors(cudaFreeHost(recv_d));

    //cout<<"    GPU Processing time: "<<GetTimer()/1e3<<"(s)"<<endl;
    return  ;
}
