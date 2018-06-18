#ifndef DATA_PROCESS_H
#define DATA_PROCESS_H

// const 
const int MAX_GPU_COUNT = 2;
const float up_value    = 100.0;

// initialize data
void DataInitial(float *fai,unsigned int size, int M, int N);

// save data
int DataSave(float *fai, int M, int N);

//dispatch total data to each node
void DispatchData(float *fai_total, float *fai_node, unsigned int dataSizePerNode, 
				  int my_rank, int comm_sz, int M, int N);

#endif