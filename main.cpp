// main.cpp 
#include <iostream>
#include <fstream>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h> 

#include "dataprocess.h"
#include "cudafunc.h"

using namespace std;



int main(int argc, char* argv[])
{
	int my_rank, comm_sz, name_len;
	unsigned int dataSizeTotal, dataSizePerNode;
	unsigned int local_M,local_N;
	char processor_name[20];

	struct timeval start,end,start2,end2;
	double timeuse,timeuse2,timeuse3,timeuse4;

	gettimeofday(&start2, NULL);

	// number of grids
	int M,N;
	if(argc==1){
		M=5120;
		N=1024;
	}
	else if(argc==3 || argc==4){
		M=atoi(argv[1]);
		N=atoi(argv[2]);
	}
	else{
		cout<<"Please input the number of grids(height,width), default: 5120 x 1024"<<endl;
		return 0;
	}

	// Initialize MPI and get process ID and the total number of processe
	MPI_Init(&argc, &argv);        
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);  

	//get the name of each node
	MPI_Get_processor_name(processor_name,&name_len);
	cout<<"process "<<my_rank<<" of "<<comm_sz<<" runs on "<<processor_name<<endl;

	//initialize data by root node
	dataSizeTotal=(M+2)*N;
	float *fai_total=NULL;
	if(my_rank==0){
		cout<<"Initializing data..."<<endl;
		fai_total = new float[dataSizeTotal];
		DataInitial(fai_total, dataSizeTotal, M, N);
	}

	gettimeofday(&start, NULL);
	//allocate memory on each node, the number of grids each node gets is local_M*local*N
	local_N=N;
	local_M=M/comm_sz+2;
	dataSizePerNode=local_N*local_M;
	float *fai_node = new float[dataSizePerNode];

	//dispatch total data to each node
	DispatchData(fai_total, fai_node, dataSizePerNode, my_rank, comm_sz, M, N);

	//calculate in GPU
	GpuCalculate(fai_node, local_M-2, local_N, my_rank, comm_sz);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather(fai_node+N, 
			   local_N*(local_M-2),
			   MPI_FLOAT,
			   fai_total+N,
			   local_N*(local_M-2),
			   MPI_FLOAT,
			   0,
			   MPI_COMM_WORLD);

	gettimeofday(&end,NULL);
	timeuse=end.tv_sec-start.tv_sec + (end.tv_usec-start.tv_usec)/1e6;

	if(argc==4){
		if(my_rank==0){
			if(DataSave(fai_total+N,M,N))
				cout<<"Data is saved successfully!"<<endl;	
		}
	}
	gettimeofday(&end2,NULL);
	timeuse2=end2.tv_sec-start2.tv_sec + (end2.tv_usec-start2.tv_usec)/1e6;
	timeuse3=end2.tv_sec-start.tv_sec + (end2.tv_usec-start.tv_usec)/1e6;
	timeuse4=start.tv_sec-start2.tv_sec + (start.tv_usec-start2.tv_usec)/1e6;

	cout<<"Calculation time used of process "<<my_rank<<" running on "<<processor_name<<" is: "<<timeuse<<"s"<<endl;
	if(my_rank==0){
		cout<<"MPI initialization time is "<<timeuse4<<"s"<<endl;
		cout<<"Total time except MPI intialization is "<<timeuse3<<"s"<<endl; 
		cout<<"Total time is "<<timeuse2<<"s"<<endl;
	}
	MPI_Finalize();
	return 0;
}
