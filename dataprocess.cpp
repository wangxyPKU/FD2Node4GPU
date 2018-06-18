// System includes
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include "dataprocess.h"

using namespace std;

/*void my_abort(int err)
{
	cout << "Test FAILED\n";
    MPI_Abort(MPI_COMM_WORLD, err);
}
void checkMPIErrors(MPIAPI)
{
	if((MPIAPI)!=MPI_SUCCESS){
		cerr<<"MPI call error! \""#function"\"\n";
		my_abort(-1); 
	}
}
*/
// initialize data
void DataInitial(float *fai,unsigned int size, int M, int N)
{
	for(int i=0; i<size; i++){
		if(i>N-1 && i<2*N)
			fai[i]=up_value;
		else
			fai[i]=0;
	}
}

// save data
int DataSave(float *fai, int M, int N)
{
	char filename[100];
	strcpy(filename,"/public/home/wang_xiaoyue/data/fai_data.txt");
	ofstream f(filename);
	if (!f) {
		cout << "File open error!" << endl;
		return 0;
	}
	for (int i = 0; i < M*N; i++) {
		f << fai[i] << ' ';
		if ((i + 1) % N == 0)
			f << endl;
	}
	f.close();
	return 1;
}

//dispatch total data to each node
void DispatchData(float *fai_total, float *fai_node, unsigned int dataSizePerNode, 
				  int my_rank, int comm_sz, int M, int N)
{
	unsigned int offset;
	MPI_Status status;
	if(my_rank==0){
		for(int dest=1; dest<comm_sz; dest++){
			offset=dest*(M/comm_sz)*N;	
			MPI_Send(fai_total+offset,
					 dataSizePerNode,
					 MPI_FLOAT,
					 dest,
					 0,
					 MPI_COMM_WORLD);
		}	
		memcpy(fai_node, fai_total, dataSizePerNode*sizeof(float));
	}
	else
		MPI_Recv(fai_node, dataSizePerNode, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
}

