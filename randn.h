#pragma once

#define PI 3.141592653


double Normal(double x,double miu,double sigma) //概率密度函数
{
	return 1.0/sqrt(2*PI*sigma)*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double AverageRandom(double min,double max)
{
	double temp;
	temp=rand();
	temp = temp/(double)RAND_MAX;//0~1
	temp=temp*(max-min)+min;//min~max
	//temp=2*max*temp;//-max~+max
	return temp;
}
double NormalRandom(double miu,double sigma,double min,double max)//产生正态分布随机数
{
	double x;
	double dScope;
	double y;
	double P;

	do
	{
		x = AverageRandom(min,max);//产生min和max之间的随机数
		y = Normal(x, miu, sigma);//x点处的概率大小y
		P=Normal(miu,miu,sigma);
		dScope = AverageRandom(0,P);//x=0点的概率大小dScope
	}while( dScope > y);
	return x;
}
float randn(int size,float *rand_N)
{
	int i;
	//char s[10];
	//string s;
	//rand_N = (float*)malloc(sizeof(float)*size);

	for(i=0;i<size;i++)
	{
		rand_N[i]=(float)NormalRandom(0,1,-6,+6);
	}
	//把数据输出到文件中

	return 0;
}
double gaussrand()
{   
	double n=0;   
	for(int i=0;i<12;i++)   
	{   
		n+=(double)rand()/RAND_MAX;   
	}   
	n=(n-6); //标准化   
	return n;
} ;
float NoCal(float dB)
{
	float SNRpbit,No_uncoded,R,No;
	//SNRpbit= exp ((dB/10)*log(10.));
	SNRpbit = pow((float)10.0,(float)(dB/10.0));
	No_uncoded=(float)1.0/SNRpbit;
	R=0.5f;
	//R = 0.4;
	No=No_uncoded/R;
	return No;
};
void V_rand(float No, int *fV, int length)
{
	double sigma;
	float *temp_fv;
	float fv;
	temp_fv = (float*)malloc(sizeof(float)*length);
	randn(length, temp_fv);
	sigma=sqrt(No/2.0);
	for (int i = 0; i<length; i++)
	{
		//fV[i] =sigma*temp_fv[i];
		fv= (1-2*fV[i]) + sigma*temp_fv[i];  //map 0->+1,1->-1
		//fV[i] *= (((float)4.0/No)*(-1));
#ifndef SOFT_MODE
		if (fv >= 0)		//hard decision
			fV[i] = 0;
		else
			fV[i] = 1;
#else
		if(fv*(1<<FRAC_NUM) - floor(fv*(1<<FRAC_NUM)) > 0.5)			//截断;
			fv = (float)(floor(fv * (1 << FRAC_NUM)) + 1) / (1 << FRAC_NUM);
		else
			fv = (float)floor(fv * (1 << FRAC_NUM)) / (1 << FRAC_NUM);
		if (fv > THRESHOLD)			//限幅;
			fv = THRESHOLD;
		else if (fv < -THRESHOLD)
			fv = -THRESHOLD;
		fV[i] = (int)(fv*(1 << FRAC_NUM));
#endif
	}
	free(temp_fv);
}

void Add_Noise(float snr)
{
	FILE  *fp_input, *fp_output;
	int *data_in1, *data_in2;

	data_in1 = (int *)malloc(sizeof(int)*TOTAL_LENGTH);
	data_in2 = (int *)malloc(sizeof(int)*TOTAL_LENGTH);

	if ((fp_input = fopen("coded.dat", "rb")) == NULL)
		printf("Open input file error!\n");

	int data;
	for (int i = 0; i < TOTAL_LENGTH; i++){
		fscanf(fp_input, "%d", &data_in1[i]);
		fscanf(fp_input, "%d", &data_in2[i]);
	}
	fclose(fp_input);


	//Add Gauss-Noise
	float no = NoCal(snr);
	V_rand(no, data_in1, TOTAL_LENGTH);
	V_rand(no, data_in2, TOTAL_LENGTH);


	if ((fp_output = fopen("coded.dat", "wb")) == NULL)
		printf("Open output file error!\n");

	for (int i = 0; i < TOTAL_LENGTH; i++){
		fprintf(fp_output, "%d %d\n", data_in1[i], data_in2[i]);
	}
	fclose(fp_output);

	free(data_in1);
	free(data_in2);

}