#pragma once

#include "totalDefine.h"

extern "C"
void CC_Encoder(int source_length)
{
	FILE *source, *coded;
	int i, j;
	int *m;
	int *c0;
	int *c1;

	m = (int *)malloc(sizeof(int)*(source_length + 7));
	c0 = (int *)malloc(sizeof(int)*source_length);
	c1 = (int *)malloc(sizeof(int)*source_length);

	source = fopen("source.dat", "wb");
	coded = fopen("coded.dat", "wb");


	//generate random bits
	for (i = 0; i < 6; i++)
		m[i] = 0;		//首6个0
	for (i = 6; i < source_length-7; i++)
		m[i] = rand() % 2;
	for (i = source_length-7; i < source_length; i++)
		m[i] = 0;		//结尾补7个0;


	//encode
	for (i = 6; i < source_length; i++)
	{
		c0[i - 6] = m[i] ^ m[i - 1] ^ m[i - 2] ^ m[i - 3] ^ m[i - 6];
		c1[i - 6] = m[i] ^ m[i - 2] ^ m[i - 3] ^ m[i - 5] ^ m[i - 6];
	}

	//store
	for (i = 0; i<source_length; i++)
	{

		//printf("%d%d\n", c0[i], c1[i]);
		fprintf(source, "%d\n", m[i]);
		fprintf(coded, "%d %d\n", c0[i], c1[i]);
	}

	free(m);
	free(c0);
	free(c1);

	fclose(source);
	fclose(coded);

}

extern "C"
void countBER(int mode, float snr)		//mode=0:GPU, mode=1:CPU
{
	FILE *fp_source, *fp_decoded, *fp_ber;

	if ((fp_source = fopen("source.dat", "rb")) == NULL)
		printf("Open source file error!\n");
	
	if (mode % 2 == 0){
		if ((fp_decoded = fopen("decoded.dat", "rb")) == NULL)
			printf("Open decoded file error!\n");
	}
	else{
		if ((fp_decoded = fopen("decoded_cpu.dat", "rb")) == NULL)
			printf("Open decoded_cpu file error!\n");
	}

	int *source, *decoded;
	source = (int *)malloc(sizeof(int)*TOTAL_LENGTH);
	decoded = (int *)malloc(sizeof(int)*TOTAL_LENGTH);

	for (int i = 0; i < TOTAL_LENGTH; i++){
		fscanf(fp_source, "%d", &source[i]);
		fscanf(fp_decoded, "%d", &decoded[i]);
	}
	fclose(fp_source);
	fclose(fp_decoded);

	int err = 0;
	for (int i = 0; i < TOTAL_LENGTH - 42; i++)
	{
		if (source[i + 5] != decoded[i])
			err++;
	}

	if (mode % 2 == 0)
	{
#ifdef PLOT_MODE
		fp_ber = fopen("BER.txt", "a+");
		fprintf(fp_ber, "%.3e\n", (float)err / TOTAL_LENGTH);
		fclose(fp_ber);
		fp_ber = NULL;
#endif
		printf("GPU Decoding: Number of error bits is %d, BER is %1.3e Mbps\n", err, (float)err / TOTAL_LENGTH);
	}
	else
	{
#ifdef PLOT_MODE
		fp_ber = fopen("BER.txt", "a+");
		fprintf(fp_ber, "%.3f %.3e ", snr, (float)err / TOTAL_LENGTH);
		fclose(fp_ber);
		fp_ber = NULL;
#endif
		printf("CPU Decoding: Number of error bits is %d, BER is %1.3e Mbps\n", err, (float)err / TOTAL_LENGTH);
	}

	free(source);
	free(decoded);

}