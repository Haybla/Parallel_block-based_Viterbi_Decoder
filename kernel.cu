#include "totalDefine.h"
#include "cuda_helper.cuh"
#include "Viterbi_GPU.cuh"
#include "CC_Encoder.h"
#include "randn.h"
#include "Viterbi_CPU.h"

#ifndef SOFT_MODE
static int h_c0[STATE_NUM][2] = {
	0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1,
	1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0,
	1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
	0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
};

static int h_c1[STATE_NUM][2] = {
	1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0,
	0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1,
	0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
	1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
};
#else
//static int h_c0[STATE_NUM][2] = {
//	32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32,
//	-32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32,
//	-32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32,
//	32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32,
//};
//
//static int h_c1[STATE_NUM][2] = {
//	-32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32,
//	32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32,
//	32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32,
//	-32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32,
//};
static int h_c0[STATE_NUM][2] = {
	-1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1,
	1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1,
	1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1,
	-1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1,
};

static int h_c1[STATE_NUM][2] = {
	1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1,
	-1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1,
	-1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1,
	1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1,
};
#endif

static int h_lookup1[STATE_NUM] = {0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 
									3, 2, 3, 2, 1, 0, 1, 0, 1, 0, 1, 0, 3, 2, 3, 2,
									0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1,
									3, 2, 3, 2, 1, 0, 1, 0, 1, 0, 1, 0, 3, 2, 3, 2};

static int h_lookup2[STATE_NUM] = { 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3,
									4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7,
									8, 8, 9, 9, 8, 8, 9, 9, 10, 10, 11, 11, 10, 10, 11, 11,
									12, 12, 13, 13, 12, 12, 13, 13, 14, 14, 15, 15, 14, 14, 15, 15};

void CC_Decoder()
{
	/*****************************************************************/
	/**********************Memory Initialization**********************/
	/*****************************************************************/
	//Host Memory
	char *data_in1;
	char *data_in2;
	FRAME *h_data_in1[STREAM_NUM];
	FRAME *h_data_in2[STREAM_NUM];
	TFRAME *th_data_in1[STREAM_NUM];
	TFRAME *th_data_in2[STREAM_NUM];
	CODE_STREAM *h_data_out[STREAM_NUM];
	TCODE_STREAM *t_h_data_out[STREAM_NUM];		//add
	int *data_out;

	//PATH *h_sur_path[STREAM_NUM];
	//PM *h_last_pm[STREAM_NUM];
	//PM *h_curr_pm[STREAM_NUM];


	//Device Memory
	FRAME *d_data_in1[STREAM_NUM];
	FRAME *d_data_in2[STREAM_NUM];
	TFRAME *td_data_in1[STREAM_NUM];
	TFRAME *td_data_in2[STREAM_NUM];
	CODE_STREAM *d_data_out[STREAM_NUM];
	TCODE_STREAM *t_d_data_out[STREAM_NUM];		//add

	PATH *d_sur_path[STREAM_NUM];
	PATH *d_sur_path_1[STREAM_NUM];		//add 20151014
	PATH *d_sur_path_2[STREAM_NUM];		//add 20151014
	PATH *d_sur_path_3[STREAM_NUM];		//add 20151014
	//TPATH *t_d_sur_path[STREAM_NUM];	//add
	//PM *d_last_pm[STREAM_NUM];
	//PM *d_curr_pm[STREAM_NUM];



	//Malloc Host Memory
	checkCudaErrors(cudaHostAlloc(&data_in1, sizeof(int)*TOTAL_LENGTH, cudaHostAllocDefault));

	checkCudaErrors(cudaHostAlloc(&data_in2, sizeof(int)*TOTAL_LENGTH, cudaHostAllocDefault));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaHostAlloc(&h_data_in1[i], sizeof(FRAME)*STREAM_COUNT, cudaHostAllocDefault));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaHostAlloc(&h_data_in2[i], sizeof(FRAME)*STREAM_COUNT, cudaHostAllocDefault));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaHostAlloc(&th_data_in1[i], sizeof(TFRAME)*(BLOCK_LENGTH + 2 * TB_LENGTH), cudaHostAllocDefault));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaHostAlloc(&th_data_in2[i], sizeof(TFRAME)*(BLOCK_LENGTH + 2 * TB_LENGTH), cudaHostAllocDefault));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaHostAlloc(&h_data_out[i], sizeof(CODE_STREAM)*STREAM_COUNT, cudaHostAllocDefault));

	for (int i = 0; i < STREAM_NUM; i++)	//add
		checkCudaErrors(cudaHostAlloc(&t_h_data_out[i], sizeof(TCODE_STREAM)*BLOCK_LENGTH, cudaHostAllocDefault));

	//checkCudaErrors(cudaHostAlloc(&data_out, sizeof(int)*TOTAL_LENGTH, cudaHostAllocDefault));


	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaHostAlloc(&h_sur_path[i], sizeof(PATH)*(BLOCK_LENGTH + 2 * TB_LENGTH), cudaHostAllocDefault));

	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaHostAlloc(&h_last_pm[i], sizeof(PM)*STREAM_COUNT, cudaHostAllocDefault));

	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaHostAlloc(&h_curr_pm[i], sizeof(PM)*STREAM_COUNT, cudaHostAllocDefault));


	//Malloc Device Memory
	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaMalloc(&d_data_in1[i], sizeof(FRAME)*STREAM_COUNT));
	
	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaMalloc(&d_data_in2[i], sizeof(FRAME)*STREAM_COUNT));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaMalloc(&td_data_in1[i], sizeof(TFRAME)*(BLOCK_LENGTH + 2 * TB_LENGTH)));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaMalloc(&td_data_in2[i], sizeof(TFRAME)*(BLOCK_LENGTH + 2 * TB_LENGTH)));
	
	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaMalloc(&d_data_out[i], sizeof(CODE_STREAM)*STREAM_COUNT));

	for (int i = 0; i < STREAM_NUM; i++)	//add
		checkCudaErrors(cudaMalloc(&t_d_data_out[i], sizeof(TCODE_STREAM)*BLOCK_LENGTH));

	for (int i = 0; i < STREAM_NUM; i++)	//change
		checkCudaErrors(cudaMalloc(&d_sur_path[i], sizeof(PATH)*(BLOCK_LENGTH + 2 * TB_LENGTH)));

	for (int i = 0; i < STREAM_NUM; i++)	//add 20151014
		checkCudaErrors(cudaMalloc(&d_sur_path_1[i], sizeof(PATH)*(BLOCK_LENGTH + 2 * TB_LENGTH)));

	for (int i = 0; i < STREAM_NUM; i++)	//add 20151014
		checkCudaErrors(cudaMalloc(&d_sur_path_2[i], sizeof(PATH)*(BLOCK_LENGTH + 2 * TB_LENGTH)));

	for (int i = 0; i < STREAM_NUM; i++)	//add 20151014
		checkCudaErrors(cudaMalloc(&d_sur_path_3[i], sizeof(PATH)*(BLOCK_LENGTH + 2 * TB_LENGTH)));

	//for (int i = 0; i < STREAM_NUM; i++)	//add
	//	checkCudaErrors(cudaMalloc(&t_d_sur_path[i], sizeof(TPATH)*(BLOCK_LENGTH + 2 * TB_LENGTH)));

	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaMalloc(&d_last_pm[i], sizeof(PM)*STREAM_COUNT));

	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaMalloc(&d_curr_pm[i], sizeof(PM)*STREAM_COUNT));


	/*****************************************************************/
	/***********************Simulation Starting***********************/
	/*****************************************************************/
	cudaEvent_t start, stop;
	float totalTime = 0;
	float testTime = 0;
	float time1 = 0, time2 = 0;


	//Matrix Copy Host to Device, Constant Memory
	checkCudaErrors(cudaMemcpyToSymbol(d_c0, h_c0, sizeof(int)*STATE_NUM*2));
	checkCudaErrors(cudaMemcpyToSymbol(d_c1, h_c1, sizeof(int)*STATE_NUM * 2));

	checkCudaErrors(cudaMemcpyToSymbol(d_lookup1, h_lookup1, sizeof(int)*STATE_NUM));
	checkCudaErrors(cudaMemcpyToSymbol(d_lookup2, h_lookup2, sizeof(int)*STATE_NUM));





	/*****************************************************************/
	/**************************GPU Decoding***************************/
	/*****************************************************************/
	//Kernel Dimension Setting
	dim3 grid(STREAM_COUNT/32);
	dim3 block(128);

	dim3 grid1(STREAM_COUNT / 32);
	dim3 block1(32);


	//Streams Creation
	cudaStream_t *str = (cudaStream_t *)malloc(STREAM_NUM * sizeof(cudaStream_t));
	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaStreamCreate(&str[i]));


	//This part should be replaced by data from demodulator
	FILE  *fp_input, *fp_output;

	if ((fp_input = fopen("coded.dat", "rb")) == NULL)
		printf("Open input file error!\n");

	if ((fp_output = fopen("decoded.dat", "wb")) == NULL)
		printf("Open output file error!\n");

	int data1, data2;
	for (int i = 0; i < TOTAL_LENGTH; i++){
		fscanf(fp_input, "%d", &data1);
		fscanf(fp_input, "%d", &data2);
		data_in1[i] = (char)data1;
		data_in2[i] = (char)data2;
	}
	fclose(fp_input);


	//convert 1 data stream to sub-blocks
	for (int i = 0; i < STREAM_NUM; i++)
	for (int j = 0; j < STREAM_COUNT; j++)
	{
		if (i == 0 && j == 0)		//the 1st block
		{
			memset(&h_data_in1[0][0][0], 0, sizeof(char)*TB_LENGTH);
			memset(&h_data_in2[0][0][0], 0, sizeof(char)*TB_LENGTH);
			memcpy(&h_data_in1[0][0][TB_LENGTH], &data_in1[0], sizeof(char)*BLOCK_LENGTH);
			memcpy(&h_data_in2[0][0][TB_LENGTH], &data_in2[0], sizeof(char)*BLOCK_LENGTH);
		}
		else
		{
			memcpy(&h_data_in1[i][j][0], &data_in1[(i*STREAM_COUNT + j)*BLOCK_LENGTH - TB_LENGTH], sizeof(FRAME));
			memcpy(&h_data_in2[i][j][0], &data_in2[(i*STREAM_COUNT + j)*BLOCK_LENGTH - TB_LENGTH], sizeof(FRAME));
		}
	}

	for (int i = 0; i < STREAM_NUM; i++)
	for (int j = 0; j < STREAM_COUNT; j++)
	for (int k = 0; k < BLOCK_LENGTH + 2 * TB_LENGTH; k++)
	{
		th_data_in1[i][k][j] = h_data_in1[i][j][k];
		th_data_in2[i][k][j] = h_data_in2[i][j][k];
	}



	//for (int i = 0; i < STREAM_NUM; i++)
	//{
	//	memset(&h_sur_path[i][0][0], 0, sizeof(PATH)*STREAM_COUNT);
	//}


	//GPU Timer
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, 0));

	//GPU Decoding Starting
	for (int str_count = 0; str_count < STREAM_NUM; str_count++)
	{
		//H2D
		//checkCudaErrors(cudaMemcpyAsync(d_data_in1[str_count], h_data_in1[str_count],
		//	sizeof(FRAME)*STREAM_COUNT, cudaMemcpyHostToDevice, str[str_count]));
		//checkCudaErrors(cudaMemcpyAsync(d_data_in2[str_count], h_data_in2[str_count],
		//	sizeof(FRAME)*STREAM_COUNT, cudaMemcpyHostToDevice, str[str_count]));
		checkCudaErrors(cudaMemcpyAsync(td_data_in1[str_count], th_data_in1[str_count],
			sizeof(TFRAME)*(BLOCK_LENGTH + 2 * TB_LENGTH), cudaMemcpyHostToDevice, str[str_count]));
		checkCudaErrors(cudaMemcpyAsync(td_data_in2[str_count], th_data_in2[str_count],
			sizeof(TFRAME)*(BLOCK_LENGTH + 2 * TB_LENGTH), cudaMemcpyHostToDevice, str[str_count]));
		//cudaMemsetAsync(&d_sur_path[str_count][0][0][0], 0, sizeof(PATH)*STREAM_COUNT, str[str_count]);

		//checkCudaErrors(cudaEventRecord(start, 0));
		
		//Kernel Execution
		Viterbi_ACS_GPU << <grid, block, 0, str[str_count] >> >(td_data_in1[str_count], td_data_in2[str_count], d_sur_path[str_count], d_sur_path_1[str_count], d_sur_path_2[str_count], d_sur_path_3[str_count]);
		
		//checkCudaErrors(cudaEventRecord(stop, 0));
		//checkCudaErrors(cudaEventSynchronize(stop));
		//checkCudaErrors(cudaEventElapsedTime(&time1, start, stop));

		//dim3 a(STREAM_COUNT, BLOCK_LENGTH + 2 * TB_LENGTH);
		//dim3 b(STATE_NUM);
		//MatrixTranspose << <a, b, 0, str[str_count] >> >(d_sur_path[str_count], t_d_sur_path[str_count]);

		//checkCudaErrors(cudaEventRecord(start, 0));

		Viterbi_Backward_GPU << <grid1, block1, 0, str[str_count] >> >(t_d_data_out[str_count], d_sur_path[str_count], d_sur_path_1[str_count], d_sur_path_2[str_count], d_sur_path_3[str_count]);

		//checkCudaErrors(cudaEventRecord(stop, 0));
		//checkCudaErrors(cudaEventSynchronize(stop));
		//checkCudaErrors(cudaEventElapsedTime(&time2, start, stop));
		//totalTime = time1 + time2;

		//D2H
		//checkCudaErrors(cudaMemcpyAsync(h_data_out[str_count], d_data_out[str_count],
		//	sizeof(CODE_STREAM)*STREAM_COUNT, cudaMemcpyDeviceToHost, str[str_count]));
		checkCudaErrors(cudaMemcpyAsync(t_h_data_out[str_count], t_d_data_out[str_count],
			sizeof(TCODE_STREAM)*BLOCK_LENGTH/8, cudaMemcpyDeviceToHost, str[str_count]));

		//checkCudaErrors(cudaMemcpyAsync(h_sur_path[str_count], d_sur_path[str_count],
		//sizeof(PATH)*(BLOCK_LENGTH + 2 * TB_LENGTH), cudaMemcpyDeviceToHost, str[str_count]));

	}

	//Cuda Streams Synchronizing
	checkCudaErrors(cudaDeviceSynchronize());


	//GPU Decoding Ending
	//FILE *sur_path = fopen("sur_path.dat", "wb");
	//for (int i = 0; i < 1; i++)
	//for (int j = 0; j < 1; j++)
	//for (int k = 0; k < BLOCK_LENGTH + 2 * TB_LENGTH; k++)
	////for (int p = 0; p < STATE_NUM; p++)		
	//	//fprintf(sur_path, "%d\n", (h_sur_path[i][k][j]>>p) & 0x1);
	//	fprintf(sur_path, "%llu\n", h_sur_path[i][k][j]);
	//fclose(sur_path);

	//GPU Timer
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&testTime, start, stop));
	totalTime += testTime;


	//store decoded data
	for (int i = 0; i < STREAM_NUM; i++)
	for (int j = 0; j < STREAM_COUNT; j++)
	for (int k = 0; k < BLOCK_LENGTH; k++)
		fprintf(fp_output, "%d\n", t_h_data_out[i][k][j]);

	fclose(fp_output);



	//Compute T/P
	printf("GPU Decoding: Total time is %1.3f ms\n", totalTime);
	printf("GPU Decoding: Thoughput is %1.3f Mbps\n", (float)TOTAL_LENGTH / totalTime / 1000);

	//printf("GPU Decoding: Time_ACS is %1.3f ms\n", time1);
	//printf("GPU Decoding: Time_Backward is %1.3f ms\n", time2);


	/*****************************************************************/
	/*************************Memory Releasing************************/
	/*****************************************************************/
	//Free Host Memory
	checkCudaErrors(cudaFreeHost(data_in1));

	checkCudaErrors(cudaFreeHost(data_in2));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFreeHost(h_data_in1[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFreeHost(h_data_in2[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFreeHost(th_data_in1[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFreeHost(th_data_in2[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFreeHost(h_data_out[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFreeHost(t_h_data_out[i]));	//add

	//checkCudaErrors(cudaFreeHost(data_out));

	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaFreeHost(h_sur_path[i]));

	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaFreeHost(h_last_pm[i]));

	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaFreeHost(h_curr_pm[i]));


	//Free Device Memory
	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFree(d_data_in1[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFree(d_data_in2[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFree(td_data_in1[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFree(td_data_in2[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFree(d_data_out[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFree(t_d_data_out[i]));	//add

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFree(d_sur_path[i]));

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFree(d_sur_path_1[i]));	//add 20151014

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFree(d_sur_path_2[i]));	//add 20151014

	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaFree(d_sur_path_3[i]));	//add 20151014


	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaFree(t_d_sur_path[i]));	//add

	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaFree(d_last_pm[i]));

	//for (int i = 0; i < STREAM_NUM; i++)
	//	checkCudaErrors(cudaFree(d_curr_pm[i]));



	//Cuda Stream Destroy
	for (int i = 0; i < STREAM_NUM; i++)
		checkCudaErrors(cudaStreamDestroy(str[i]));


	//Cuda Event Destroy
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	//GPU Device Reset
	checkCudaErrors(cudaDeviceReset());

	//Exit

}


int main()
{
	/*****************************************************************/
	/********************GPU Device Initialization********************/
	/*****************************************************************/

	checkCudaErrors(cudaDeviceReset());

	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	checkCudaErrors(cudaSetDevice(0));

	int deviceCount = 0;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("Device %d: \"%s\"\n", dev, deviceProp.name);
		/*****************************************************************/
		/********************Simulation Initialization********************/
		/*****************************************************************/
		//Matrix Initialization


		//Random Seed Initialization
		//srand((unsigned)time(NULL));
		srand(1);

		//Simulation Parameters Setting: SNR, cuda_streams, code_streams
#ifdef PLOT_MODE
		for (float snr = 0; snr <= 4.5; snr += 0.5)
		{
#else
		float snr = 4.0f;
#endif
		printf("Simulation SNR: %1.3f dB\n", snr);

		printf("Number of total bits is %1.3f Kb\n", (float)TOTAL_LENGTH / 1024);

		//Encoding
		CC_Encoder(TOTAL_LENGTH);

		//Adding Gauss-Noise
		Add_Noise(snr);

		//Decoding
		Viterbi_CPU();
		CC_Decoder();

		//Calculate BER
		//countBER(1, snr);	//CPU decoding
		//countBER(0, snr);	//GPU decoding

		printf("--------------------------------------------------------------------\n\n");
#ifdef PLOT_MODE
		}
#endif
	}
	exit(EXIT_SUCCESS);
}


