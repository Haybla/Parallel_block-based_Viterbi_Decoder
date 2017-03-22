#pragma once

#include "totalDefine.h"

#define  state_num    64
#define  tblen        42

void  Viterbi_CPU()
{

	/*path0 and path1 output  value*/
#ifndef SOFT_MODE
	static  int  c0[state_num][2] = {
		0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1,
		1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0,
		1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
		0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
	};

	static  int  c1[state_num][2] = {
		1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0,
		0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1,
		0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
		1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
	};
#else
	static int c0[state_num][2] = {
		32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32,
		-32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32,
		-32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32,
		32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32,
	};
	
	static int c1[state_num][2] = {
		-32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32,
		32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32,
		32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32, -32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32,
		-32, 32, 32, -32, -32, -32, 32, 32, -32, 32, 32, -32, -32, -32, 32, 32, 32, -32, -32, 32, 32, 32, -32, -32, 32, -32, -32, 32, 32, 32, -32, -32,
	};
	//static  int  c0[state_num][2] = {
	//	-1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1,
	//	1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1,
	//	1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1,
	//	-1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1,
	//};
	//static  int  c1[state_num][2] = {
	//	1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1,
	//	-1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1,
	//	-1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1,
	//	1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1,
	//};
#endif


	static  unsigned  int  sur_path[state_num][tblen];
	static  int  last_pm[state_num], curr_pm[state_num];

	static  unsigned  int  decode[tblen];

	int  data_in1, data_in2;

	int  j, k = 0, m, n, x, y, f;
	int  mm = 0;
	int  distant00, distant01, distant10, distant11, BM00, BM01, BM10, BM11;

	int  tb_en = 0;



	int  sur_row, sur_column;
	int minstate = 0;
	unsigned	int  state[6], num = 0, start_state = 0;


	FILE  *fp_input, *fp_output;
	int   data;

	if ((fp_input = fopen("coded.dat", "rb")) == NULL)
		printf("Open  input file   error!\n");

	if ((fp_output = fopen("decoded_cpu.dat", "wb")) == NULL)
		printf("Open  output file  error!\n");

	clock_t start, stop;

	start = clock();

	while (!feof(fp_input)){
		fscanf(fp_input, "%d", &data_in1);
		fscanf(fp_input, "%d", &data_in2);


		k = k + 1;
		mm = mm + 1;
		if (k >= tblen){
			tb_en = 1;
			k = tblen;
		}
		else{
			tb_en = 0;
		}


		/******Begin  add-compare-select ****/

		for (j = 0; j <= 31; j++){
#ifndef SOFT_MODE
			distant00=(abs(data_in1-c0[2*j][0])) +abs(data_in2-c0[2*j][1]);
			distant01=(abs(data_in1-c0[2*j+1][0]))+abs(data_in2-c0[2*j+1][1]);
			distant10=(abs(data_in1-c1[2*j][0]))+abs(data_in2-c1[2*j][1]);
			distant11=(abs(data_in1-c1[2*j+1][0]))+abs(data_in2-c1[2*j+1][1]);
#else
			distant00 = (data_in1 - c0[2 * j][0])*(data_in1 - c0[2 * j][0]) + (data_in2 - c0[2 * j][1])*(data_in2 - c0[2 * j][1]);
			distant01 = (data_in1 - c0[2 * j + 1][0])*(data_in1 - c0[2 * j + 1][0]) + (data_in2 - c0[2 * j + 1][1])*(data_in2 - c0[2 * j + 1][1]);
			distant10 = (data_in1 - c1[2 * j][0])*(data_in1 - c1[2 * j][0]) + (data_in2 - c1[2 * j][1])*(data_in2 - c1[2 * j][1]);
			distant11 = (data_in1 - c1[2 * j + 1][0])*(data_in1 - c1[2 * j + 1][0]) + (data_in2 - c1[2 * j + 1][1])*(data_in2 - c1[2 * j + 1][1]);

			//distant00 = data_in1 * c0[2 * j][0] + data_in2 * c0[2 * j][1];
			//distant01 = data_in1 * c0[2 * j + 1][0] + data_in2 * c0[2 * j + 1][1];
			//distant10 = data_in1 * c1[2 * j][0] + data_in2 * c1[2 * j][1];
			//distant11 = data_in1 * c1[2 * j + 1][0] + data_in2 * c1[2 * j + 1][1];
#endif


			BM00 = last_pm[2 * j] + distant00;
			BM01 = last_pm[2 * j + 1] + distant01;
			BM10 = last_pm[2 * j] + distant10;
			BM11 = last_pm[2 * j + 1] + distant11;


			if (BM00<BM01){
				sur_path[j][k - 1] = 0;
				curr_pm[j] = BM00;
			}
			else{
				sur_path[j][k - 1] = 1;
				curr_pm[j] = BM01;
			}



			if (BM10<BM11){
				sur_path[j + 32][k - 1] = 0;
				curr_pm[j + 32] = BM10;
			}
			else{
				sur_path[j + 32][k - 1] = 1;
				curr_pm[j + 32] = BM11;
			}

		}


		/******check  data  flow  ******/
		minstate = curr_pm[0];
		for (m = 0; m <= 63; m++){
			if (curr_pm[m] <= minstate){
				minstate = curr_pm[m];
				num = m;
				start_state = m;
			}
		}


		for (f = 0; f <= 63; f++)
			last_pm[f] = curr_pm[f] - minstate;


		if (tb_en == 1){
			for (n = 0; n <= 5; n++){
				if (num >= pow(2.0, (5 - n))){
					state[n] = 1;
					num = num - pow(2.0, (5 - n));
				}
				else
					state[n] = 0;
			}

			for (x = tblen - 1; x >= 0; x--){

				for (y = 0; y <= 4; y++)state[y] = state[y + 1];
				state[5] = sur_path[start_state][x];
				start_state = state[0] * 32 + state[1] * 16 + state[2] * 8 + state[3] * 4 + state[4] * 2 + state[5] * 1;
			}

			fprintf(fp_output, "%d\n", state[0]);
			//printf("%d\n", state[0]);

			for (sur_row = 0; sur_row <= 63; sur_row++){
				for (sur_column = 0; sur_column <= tblen - 2; sur_column++){
					sur_path[sur_row][sur_column] = sur_path[sur_row][sur_column + 1];
				}
			}
		}
	}

	stop = clock();

	double time = ((double)(stop - start) / CLK_TCK);

	//printf("CPU Decoding: Thoughput is %1.3f Mbps\n", TOTAL_LENGTH/time/1e6);


	fclose(fp_input);
	fclose(fp_output);
}