#pragma once

#include "totalDefine.h"


/*1 warp 1 block version, sur_path uses GMEM*/
__global__ void Viterbi_ACS_GPU(TFRAME *d_data_in1, TFRAME *d_data_in2, PATH *sur_path, PATH *sur_path_1, PATH *sur_path_2, PATH *sur_path_3)
{
	/*
	1 threads ↔ 1 codeblock, 
	64 threads ↔ 1 threadblock,
	so 1 threadblock ↔ 64 codeblocks,

	P = codeblock.cpos
	tid = thread.id
	Y*X + P = codeblock.pos
	*/

	int CLS = threadIdx.x / 32;	//0~63
	int P = threadIdx.x % 32;		//0~3
	int Y = blockIdx.x;
	int X = 32;		//64

	__shared__ PM curr_pm[STATE_NUM];
	//__shared__ uint64 sur[64];

	register int i, j;
	register int data_in1, data_in2;
	register int distance0, distance1; 
	register int BM0, BM1, BM2, BM3;
	register int PM00, PM01, PM10, PM11, PM20, PM21, PM30, PM31;
	register int PM40, PM41, PM50, PM51, PM60, PM61, PM70, PM71,

	register int data;
	register ushort su = 0;

	register int cnt = 0;

	while (cnt < (BLOCK_LENGTH + 2*TB_LENGTH))
	{
		data_in1 = (int)d_data_in1[cnt][Y*X + P];
		data_in2 = (int)d_data_in2[cnt][Y*X + P];

		//data_in1 = 1;
		//data_in2 = 1;

		cnt++;

		su = 0;

		if (CLS == 0)	//class A
		{
			distance0 = data_in1 * -1 + data_in2 * -1;
			distance1 = data_in1 * 1 + data_in2 * 1;

			//s0	s0
			//	  *
			//s1	s32
			BM0 = curr_pm[0][P] + distance0;
			BM1 = curr_pm[0][P] + distance1;
			BM2 = curr_pm[1][P] + distance1;
			BM3 = curr_pm[1][P] + distance0;
			if (BM0 < BM2){
				PM00 = BM0;
			}
			else{
				PM00 = BM2;
				su += ((ushort)1);
			}
			if (BM1 < BM3){
				PM01 = BM1;
			}
			else{
				PM01 = BM3;
				su += ((ushort)1) << 8;
			}

			//s4	s2
			//	  *
			//s5	s34
			BM0 = curr_pm[4][P] + distance0;
			BM1 = curr_pm[4][P] + distance1;
			BM2 = curr_pm[5][P] + distance1;
			BM3 = curr_pm[5][P] + distance0;
			if (BM0 < BM2){
				PM10 = BM0;
			}
			else{
				PM10 = BM2;
				su += ((ushort)1) << 1;
			}
			if (BM1 < BM3){
				PM11 = BM1;
			}
			else{
				PM11 = BM3;
				su += ((ushort)1) << 9;
			}

			//s24	s12
			//	  *
			//s25	s44
			BM0 = curr_pm[24][P] + distance0;
			BM1 = curr_pm[24][P] + distance1;
			BM2 = curr_pm[25][P] + distance1;
			BM3 = curr_pm[25][P] + distance0;
			if (BM0 < BM2){
				PM20 = BM0;
			}
			else{
				PM20 = BM2;
				su += ((ushort)1) << 2;
			}
			if (BM1 < BM3){
				PM21 = BM1;
			}
			else{
				PM21 = BM3;
				su += ((ushort)1) << 10;
			}

			//s28	s14
			//	  *
			//s29	s46
			BM0 = curr_pm[28][P] + distance0;
			BM1 = curr_pm[28][P] + distance1;
			BM2 = curr_pm[29][P] + distance1;
			BM3 = curr_pm[29][P] + distance0;
			if (BM0 < BM2){
				PM30 = BM0;
			}
			else{
				PM30 = BM2;
				su += ((ushort)1) << 3;
			}
			if (BM1 < BM3){
				PM31 = BM1;
			}
			else{
				PM31 = BM3;
				su += ((ushort)1) << 11;
			}

			//s42	s21
			//	  *
			//s43	s53
			BM0 = curr_pm[42][P] + distance0;
			BM1 = curr_pm[42][P] + distance1;
			BM2 = curr_pm[43][P] + distance1;
			BM3 = curr_pm[43][P] + distance0;
			if (BM0 < BM2){
				PM40 = BM0;
			}
			else{
				PM40 = BM2;
				su += ((ushort)1) << 4;
			}
			if (BM1 < BM3){
				PM41 = BM1;
			}
			else{
				PM41 = BM3;
				su += ((ushort)1) << 12;
			}

			//s46	s23
			//	  *
			//s47	s55
			BM0 = curr_pm[46][P] + distance0;
			BM1 = curr_pm[46][P] + distance1;
			BM2 = curr_pm[47][P] + distance1;
			BM3 = curr_pm[47][P] + distance0;
			if (BM0 < BM2){
				PM50 = BM0;
			}
			else{
				PM50 = BM2;
				su += ((ushort)1) << 5;
			}
			if (BM1 < BM3){
				PM51 = BM1;
			}
			else{
				PM51 = BM3;
				su += ((ushort)1) << 13;
			}

			//s50	s25
			//	  *
			//s51	s57
			BM0 = curr_pm[50][P] + distance0;
			BM1 = curr_pm[50][P] + distance1;
			BM2 = curr_pm[51][P] + distance1;
			BM3 = curr_pm[51][P] + distance0;
			if (BM0 < BM2){
				PM60 = BM0;
			}
			else{
				PM60 = BM2;
				su += ((ushort)1) << 6;
			}
			if (BM1 < BM3){
				PM61 = BM1;
			}
			else{
				PM61 = BM3;
				su += ((ushort)1) << 14;
			}

			//s54	s27
			//	  *
			//s55	s59
			BM0 = curr_pm[54][P] + distance0;
			BM1 = curr_pm[54][P] + distance1;
			BM2 = curr_pm[55][P] + distance1;
			BM3 = curr_pm[55][P] + distance0;
			if (BM0 < BM2){
				PM70 = BM0;
			}
			else{
				PM70 = BM2;
				su += ((ushort)1) << 7;
			}
			if (BM1 < BM3){
				PM71 = BM1;
			}
			else{
				PM71 = BM3;
				su += ((ushort)1) << 15;
			}

			__syncthreads();

			curr_pm[0][P] = PM00;
			curr_pm[32][P] = PM01;
			curr_pm[2][P] = PM10;
			curr_pm[34][P] = PM11;
			curr_pm[12][P] = PM20;
			curr_pm[44][P] = PM21;
			curr_pm[14][P] = PM30;
			curr_pm[46][P] = PM31;
			curr_pm[21][P] = PM40;
			curr_pm[53][P] = PM41;
			curr_pm[23][P] = PM50;
			curr_pm[55][P] = PM51;
			curr_pm[25][P] = PM60;
			curr_pm[57][P] = PM61;
			curr_pm[27][P] = PM70;
			curr_pm[59][P] = PM71;

			sur_path[cnt - 1][0][Y*X + P] = su;

			__syncthreads();
		}

		if (CLS == 1)	//class B
		{
			distance0 = data_in1 * -1 + data_in2 * 1;
			distance1 = data_in1 * 1 + data_in2 * -1;

			//s2	s1
			//	  *
			//s3	s33
			BM0 = curr_pm[2][P] + distance0;
			BM1 = curr_pm[2][P] + distance1;
			BM2 = curr_pm[3][P] + distance1;
			BM3 = curr_pm[3][P] + distance0;
			if (BM0 < BM2){
				PM00 = BM0;
			}
			else{
				PM00 = BM2;
				su += ((ushort)1);
			}
			if (BM1 < BM3){
				PM01 = BM1;
			}
			else{
				PM01 = BM3;
				su += ((ushort)1) << 8;
			}

			//s6	s3
			//	  *
			//s7	s35
			BM0 = curr_pm[6][P] + distance0;
			BM1 = curr_pm[6][P] + distance1;
			BM2 = curr_pm[7][P] + distance1;
			BM3 = curr_pm[7][P] + distance0;
			if (BM0 < BM2){
				PM10 = BM0;
			}
			else{
				PM10 = BM2;
				su += ((ushort)1) << 1;
			}
			if (BM1 < BM3){
				PM11 = BM1;
			}
			else{
				PM11 = BM3;
				su += ((ushort)1) << 9;
			}

			//s26	s13
			//	  *
			//s27	s45
			BM0 = curr_pm[26][P] + distance0;
			BM1 = curr_pm[26][P] + distance1;
			BM2 = curr_pm[27][P] + distance1;
			BM3 = curr_pm[27][P] + distance0;
			if (BM0 < BM2){
				PM20 = BM0;
			}
			else{
				PM20 = BM2;
				su += ((ushort)1) << 2;
			}
			if (BM1 < BM3){
				PM21 = BM1;
			}
			else{
				PM21 = BM3;
				su += ((ushort)1) << 10;
			}

			//s30	s15
			//	  *
			//s31	s47
			BM0 = curr_pm[30][P] + distance0;
			BM1 = curr_pm[30][P] + distance1;
			BM2 = curr_pm[31][P] + distance1;
			BM3 = curr_pm[31][P] + distance0;
			if (BM0 < BM2){
				PM30 = BM0;
			}
			else{
				PM30 = BM2;
				su += ((ushort)1) << 3;
			}
			if (BM1 < BM3){
				PM31 = BM1;
			}
			else{
				PM31 = BM3;
				su += ((ushort)1) << 11;
			}

			//s40	s20
			//	  *
			//s41	s52
			BM0 = curr_pm[40][P] + distance0;
			BM1 = curr_pm[40][P] + distance1;
			BM2 = curr_pm[41][P] + distance1;
			BM3 = curr_pm[41][P] + distance0;
			if (BM0 < BM2){
				PM40 = BM0;
			}
			else{
				PM40 = BM2;
				su += ((ushort)1) << 4;
			}
			if (BM1 < BM3){
				PM41 = BM1;
			}
			else{
				PM41 = BM3;
				su += ((ushort)1) << 12;
			}

			//s44	s22
			//	  *
			//s45	s54
			BM0 = curr_pm[44][P] + distance0;
			BM1 = curr_pm[44][P] + distance1;
			BM2 = curr_pm[45][P] + distance1;
			BM3 = curr_pm[45][P] + distance0;
			if (BM0 < BM2){
				PM50 = BM0;
			}
			else{
				PM50 = BM2;
				su += ((ushort)1) << 5;
			}
			if (BM1 < BM3){
				PM51 = BM1;
			}
			else{
				PM51 = BM3;
				su += ((ushort)1) << 13;
			}

			//s48	s24
			//	  *
			//s49	s56
			BM0 = curr_pm[48][P] + distance0;
			BM1 = curr_pm[48][P] + distance1;
			BM2 = curr_pm[49][P] + distance1;
			BM3 = curr_pm[49][P] + distance0;
			if (BM0 < BM2){
				PM60 = BM0;
			}
			else{
				PM60 = BM2;
				su += ((ushort)1) << 6;
			}
			if (BM1 < BM3){
				PM61 = BM1;
			}
			else{
				PM61 = BM3;
				su += ((ushort)1) << 14;
			}

			//s52	s26
			//	  *
			//s53	s58
			BM0 = curr_pm[52][P] + distance0;
			BM1 = curr_pm[52][P] + distance1;
			BM2 = curr_pm[53][P] + distance1;
			BM3 = curr_pm[53][P] + distance0;
			if (BM0 < BM2){
				PM70 = BM0;
			}
			else{
				PM70 = BM2;
				su += ((ushort)1) << 7;
			}
			if (BM1 < BM3){
				PM71 = BM1;
			}
			else{
				PM71 = BM3;
				su += ((ushort)1) << 15;
			}

			__syncthreads();

			curr_pm[1][P] = PM00;
			curr_pm[33][P] = PM01;
			curr_pm[3][P] = PM10;
			curr_pm[35][P] = PM11;
			curr_pm[13][P] = PM20;
			curr_pm[45][P] = PM21;
			curr_pm[15][P] = PM30;
			curr_pm[47][P] = PM31;
			curr_pm[20][P] = PM40;
			curr_pm[52][P] = PM41;
			curr_pm[22][P] = PM50;
			curr_pm[54][P] = PM51;
			curr_pm[24][P] = PM60;
			curr_pm[56][P] = PM61;
			curr_pm[26][P] = PM70;
			curr_pm[58][P] = PM71;

			sur_path[cnt - 1][1][Y*X + P] = su;

			__syncthreads();
		}

		if (CLS == 2)	//class C
		{
			distance0 = data_in1 * 1 + data_in2 * 1;
			distance1 = data_in1 * -1 + data_in2 * -1;

			//s8	s4
			//	  *
			//s9	s36
			BM0 = curr_pm[8][P] + distance0;
			BM1 = curr_pm[8][P] + distance1;
			BM2 = curr_pm[9][P] + distance1;
			BM3 = curr_pm[9][P] + distance0;
			if (BM0 < BM2){
				PM00 = BM0;
			}
			else{
				PM00 = BM2;
				su += ((ushort)1);
			}
			if (BM1 < BM3){
				PM01 = BM1;
			}
			else{
				PM01 = BM3;
				su += ((ushort)1) << 8;
			}

			//s12	s6
			//	  *
			//s13	s38
			BM0 = curr_pm[12][P] + distance0;
			BM1 = curr_pm[12][P] + distance1;
			BM2 = curr_pm[13][P] + distance1;
			BM3 = curr_pm[13][P] + distance0;
			if (BM0 < BM2){
				PM10 = BM0;
			}
			else{
				PM10 = BM2;
				su += ((ushort)1) << 1;
			}
			if (BM1 < BM3){
				PM11 = BM1;
			}
			else{
				PM11 = BM3;
				su += ((ushort)1) << 9;
			}

			//s16	s8
			//	  *
			//s17	s40
			BM0 = curr_pm[16][P] + distance0;
			BM1 = curr_pm[16][P] + distance1;
			BM2 = curr_pm[17][P] + distance1;
			BM3 = curr_pm[17][P] + distance0;
			if (BM0 < BM2){
				PM20 = BM0;
			}
			else{
				PM20 = BM2;
				su += ((ushort)1) << 2;
			}
			if (BM1 < BM3){
				PM21 = BM1;
			}
			else{
				PM21 = BM3;
				su += ((ushort)1) << 10;
			}

			//s20	s10
			//	  *
			//s21	s42
			BM0 = curr_pm[20][P] + distance0;
			BM1 = curr_pm[20][P] + distance1;
			BM2 = curr_pm[21][P] + distance1;
			BM3 = curr_pm[21][P] + distance0;
			if (BM0 < BM2){
				PM30 = BM0;
			}
			else{
				PM30 = BM2;
				su += ((ushort)1) << 3;
			}
			if (BM1 < BM3){
				PM31 = BM1;
			}
			else{
				PM31 = BM3;
				su += ((ushort)1) << 11;
			}

			//s34	s17
			//	  *
			//s35	s49
			BM0 = curr_pm[34][P] + distance0;
			BM1 = curr_pm[34][P] + distance1;
			BM2 = curr_pm[35][P] + distance1;
			BM3 = curr_pm[35][P] + distance0;
			if (BM0 < BM2){
				PM40 = BM0;
			}
			else{
				PM40 = BM2;
				su += ((ushort)1) << 4;
			}
			if (BM1 < BM3){
				PM41 = BM1;
			}
			else{
				PM41 = BM3;
				su += ((ushort)1) << 12;
			}

			//s38	s19
			//	  *
			//s39	s51
			BM0 = curr_pm[38][P] + distance0;
			BM1 = curr_pm[38][P] + distance1;
			BM2 = curr_pm[39][P] + distance1;
			BM3 = curr_pm[39][P] + distance0;
			if (BM0 < BM2){
				PM50 = BM0;
			}
			else{
				PM50 = BM2;
				su += ((ushort)1) << 5;
			}
			if (BM1 < BM3){
				PM51 = BM1;
			}
			else{
				PM51 = BM3;
				su += ((ushort)1) << 13;
			}

			//s58	s29
			//	  *
			//s59	s61
			BM0 = curr_pm[58][P] + distance0;
			BM1 = curr_pm[58][P] + distance1;
			BM2 = curr_pm[59][P] + distance1;
			BM3 = curr_pm[59][P] + distance0;
			if (BM0 < BM2){
				PM60 = BM0;
			}
			else{
				PM60 = BM2;
				su += ((ushort)1) << 6;
			}
			if (BM1 < BM3){
				PM61 = BM1;
			}
			else{
				PM61 = BM3;
				su += ((ushort)1) << 14;
			}

			//s62	s31
			//	  *
			//s63	s63
			BM0 = curr_pm[62][P] + distance0;
			BM1 = curr_pm[62][P] + distance1;
			BM2 = curr_pm[63][P] + distance1;
			BM3 = curr_pm[63][P] + distance0;
			if (BM0 < BM2){
				PM70 = BM0;
			}
			else{
				PM70 = BM2;
				su += ((ushort)1) << 7;
			}
			if (BM1 < BM3){
				PM71 = BM1;
			}
			else{
				PM71 = BM3;
				su += ((ushort)1) << 15;
			}

			__syncthreads();

			curr_pm[4][P] = PM00;
			curr_pm[36][P] = PM01;
			curr_pm[6][P] = PM10;
			curr_pm[38][P] = PM11;
			curr_pm[8][P] = PM20;
			curr_pm[40][P] = PM21;
			curr_pm[10][P] = PM30;
			curr_pm[42][P] = PM31;
			curr_pm[17][P] = PM40;
			curr_pm[49][P] = PM41;
			curr_pm[19][P] = PM50;
			curr_pm[51][P] = PM51;
			curr_pm[29][P] = PM60;
			curr_pm[61][P] = PM61;
			curr_pm[31][P] = PM70;
			curr_pm[63][P] = PM71;

			sur_path[cnt - 1][2][Y*X + P] = su;

			__syncthreads();
		}

		if (CLS == 3)	//class D
		{
			distance0 = data_in1 * 1 + data_in2 * -1;
			distance1 = data_in1 * -1 + data_in2 * 1;

			//s10	s5
			//	  *
			//s11	s37
			BM0 = curr_pm[10][P] + distance0;
			BM1 = curr_pm[10][P] + distance1;
			BM2 = curr_pm[11][P] + distance1;
			BM3 = curr_pm[11][P] + distance0;
			if (BM0 < BM2){
				PM00 = BM0;
			}
			else{
				PM00 = BM2;
				su += ((ushort)1);
			}
			if (BM1 < BM3){
				PM01 = BM1;
			}
			else{
				PM01 = BM3;
				su += ((ushort)1) << 8;
			}

			//s14	s7
			//	  *
			//s15	s39
			BM0 = curr_pm[14][P] + distance0;
			BM1 = curr_pm[14][P] + distance1;
			BM2 = curr_pm[15][P] + distance1;
			BM3 = curr_pm[15][P] + distance0;
			if (BM0 < BM2){
				PM10 = BM0;
			}
			else{
				PM10 = BM2;
				su += ((ushort)1) << 1;
			}
			if (BM1 < BM3){
				PM11 = BM1;
			}
			else{
				PM11 = BM3;
				su += ((ushort)1) << 9;
			}

			//s18	s9
			//	  *
			//s19	s41
			BM0 = curr_pm[18][P] + distance0;
			BM1 = curr_pm[18][P] + distance1;
			BM2 = curr_pm[19][P] + distance1;
			BM3 = curr_pm[19][P] + distance0;
			if (BM0 < BM2){
				PM20 = BM0;
			}
			else{
				PM20 = BM2;
				su += ((ushort)1) << 2;
			}
			if (BM1 < BM3){
				PM21 = BM1;
			}
			else{
				PM21 = BM3;
				su += ((ushort)1) << 10;
			}

			//s22	s11
			//	  *
			//s23	s43
			BM0 = curr_pm[22][P] + distance0;
			BM1 = curr_pm[22][P] + distance1;
			BM2 = curr_pm[23][P] + distance1;
			BM3 = curr_pm[23][P] + distance0;
			if (BM0 < BM2){
				PM30 = BM0;
			}
			else{
				PM30 = BM2;
				su += ((ushort)1) << 3;
			}
			if (BM1 < BM3){
				PM31 = BM1;
			}
			else{
				PM31 = BM3;
				su += ((ushort)1) << 11;
			}

			//s32	s16
			//	  *
			//s33	s48
			BM0 = curr_pm[32][P] + distance0;
			BM1 = curr_pm[32][P] + distance1;
			BM2 = curr_pm[33][P] + distance1;
			BM3 = curr_pm[33][P] + distance0;
			if (BM0 < BM2){
				PM40 = BM0;
			}
			else{
				PM40 = BM2;
				su += ((ushort)1) << 4;
			}
			if (BM1 < BM3){
				PM41 = BM1;
			}
			else{
				PM41 = BM3;
				su += ((ushort)1) << 12;
			}

			//s36	s18
			//	  *
			//s37	s50
			BM0 = curr_pm[36][P] + distance0;
			BM1 = curr_pm[36][P] + distance1;
			BM2 = curr_pm[37][P] + distance1;
			BM3 = curr_pm[37][P] + distance0;
			if (BM0 < BM2){
				PM50 = BM0;
			}
			else{
				PM50 = BM2;
				su += ((ushort)1) << 5;
			}
			if (BM1 < BM3){
				PM51 = BM1;
			}
			else{
				PM51 = BM3;
				su += ((ushort)1) << 13;
			}

			//s56	s28
			//	  *
			//s57	s60
			BM0 = curr_pm[56][P] + distance0;
			BM1 = curr_pm[56][P] + distance1;
			BM2 = curr_pm[57][P] + distance1;
			BM3 = curr_pm[57][P] + distance0;
			if (BM0 < BM2){
				PM60 = BM0;
			}
			else{
				PM60 = BM2;
				su += ((ushort)1) << 6;
			}
			if (BM1 < BM3){
				PM61 = BM1;
			}
			else{
				PM61 = BM3;
				su += ((ushort)1) << 14;
			}

			//s60	s30
			//	  *
			//s61	s62
			BM0 = curr_pm[60][P] + distance0;
			BM1 = curr_pm[60][P] + distance1;
			BM2 = curr_pm[61][P] + distance1;
			BM3 = curr_pm[61][P] + distance0;
			if (BM0 < BM2){
				PM70 = BM0;
			}
			else{
				PM70 = BM2;
				su += ((ushort)1) << 7;
			}
			if (BM1 < BM3){
				PM71 = BM1;
			}
			else{
				PM71 = BM3;
				su += ((ushort)1) << 15;
			}

			__syncthreads();

			curr_pm[5][P] = PM00;
			curr_pm[37][P] = PM01;
			curr_pm[7][P] = PM10;
			curr_pm[39][P] = PM11;
			curr_pm[9][P] = PM20;
			curr_pm[41][P] = PM21;
			curr_pm[11][P] = PM30;
			curr_pm[43][P] = PM31;
			curr_pm[16][P] = PM40;
			curr_pm[48][P] = PM41;
			curr_pm[18][P] = PM50;
			curr_pm[50][P] = PM51;
			curr_pm[28][P] = PM60;
			curr_pm[60][P] = PM61;
			curr_pm[30][P] = PM70;
			curr_pm[62][P] = PM71;

			sur_path[cnt - 1][3][Y*X + P] = su;

			__syncthreads();
		}




		//Reduction

		//8-4
		//sur[P] = su;

		//4-2
		//if (tid < 2)
		//{
		//	sur[P][tid] += sur[P][tid + 2];		//0.3
		//}

		//if (tid == 0)
		//{
			//2-1
			//sur[P][tid] += sur[P][tid + 1];		//0.86
			//sur_path[cnt - 1][Y*X + P] = su;
			//sur_path[cnt - 1][Y*X + P] = sur[P][tid];
			//sur_path_1[cnt - 1][Y*X + P] = sur[P][j+1];
			//sur_path_2[k - 1][Y*X + P] = sur[P][j + 2];
			//sur_path_3[k - 1][Y*X + P] = sur[P][j + 3];

		//}


	}


}

__global__ void Viterbi_Backward_GPU(TCODE_STREAM *d_data_out, PATH *sur_path, PATH *sur_path_1, PATH *sur_path_2, PATH *sur_path_3)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	uchar state = 0;
	uchar state1 = 0;
	//uchar state0 = 0, state1 = 0, state2 = 0, state3 = 0, state4 = 0, state5 = 0;

	int x;
	ushort su, su1, su2, su3;

	for (x = (BLOCK_LENGTH + 2 * TB_LENGTH - 1); x >= TB_LENGTH; x--)
	{
		//state0 = state1;
		//state1 = state2;
		//state2 = state3;
		//state3 = state4;
		//state4 = state5;

		su = sur_path[x][d_lookup1[state]][tid];
		//su1 = sur_path_1[x][d_lookup1[state]][tid];
		//su2 = sur_path_2[x][d_lookup1[state]][tid];
		//su3 = sur_path_3[x][d_lookup1[state]][tid];

		//uint64 sur1 = sur_path_1[x][tid];
		//uint64 sur2 = sur_path_2[x][tid];
		//uint64 sur3 = sur_path_3[x][tid];

		//su = su + su1 + su2 + su3;

		state1 = state;

		//state = ((state << 1) & 0x3e) + ((su >> state) & 0x1);
		state = ((state << 1) & 0x3e) + ((su>>d_lookup2[state]) & 0x1);
		//state5 = sur_path[tid][x][state];
		//state5 = sur_path[x][state][tid];

		//state = state0 * 32 + state1 * 16 + state2 * 8 + state3 * 4 + state4 * 2 + state5 * 1;
		
		if (x <= (BLOCK_LENGTH + TB_LENGTH - 1))
			d_data_out[x - TB_LENGTH][tid] = (state >> 5) & 0x01;
			//d_data_out[tid][x - TB_LENGTH] = (state>>5) & 0x01;
	}

}

/*__global__ void MatrixTranspose(PATH *sur_path, TPATH *t_sur_path)
{
	int tid = blockIdx.x;		//STREAM_COUNT
	int x = blockIdx.y;			//BLOCK_LENGTH + 2*TB_LENGTH
	int state = threadIdx.x;	//STATE_NUM

	//uchar data;
	//data = sur_path[tid][x][state];

	t_sur_path[x][state][tid] = sur_path[tid][x];

}*/