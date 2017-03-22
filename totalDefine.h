#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "malloc.h"
#include "string.h"
#include "cuda_runtime.h"
#include "math.h"
#include "time.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda_device_runtime_api.h"
#include <curand.h>
#include <curand_kernel.h>
#include <Windows.h>

#define SOFT_MODE		//soft viterbi mode


#ifdef SOFT_MODE
#define FRAC_NUM 4
#define THRESHOLD 7.9375
#endif

//#define PLOT_MODE

#define STATE_NUM 64		//number of states
#define STREAM_NUM 3		//number of async streams
#define STREAM_COUNT 10000		//number of frames in each async streams
#define BLOCK_LENGTH 512		//number of bits in frame
#define TB_LENGTH 42			//length of tail-bite
#define TOTAL_LENGTH (STREAM_NUM*STREAM_COUNT*BLOCK_LENGTH)		//message length


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned long long int uint64;

//typedef uchar PATH_SUB[STATE_NUM];
typedef ushort PATH_SUB[STREAM_COUNT];
typedef PATH_SUB PATH[4];	//change

//typedef uchar TPATH_SUB[STREAM_COUNT];	//add
//typedef TPATH_SUB TPATH[STATE_NUM];


#ifdef SOFT_MODE
typedef int PM[32];
#else
typedef ushort PM[STATE_NUM];
#endif

typedef int CODE_STREAM_P[STREAM_COUNT];
typedef int CODE_STREAM[BLOCK_LENGTH];

typedef char TCODE_STREAM[STREAM_COUNT];	//add

typedef char FRAME[BLOCK_LENGTH + 2*TB_LENGTH];
typedef char TFRAME[STREAM_COUNT];

typedef uint STATE[6];

typedef uchar DECODE[BLOCK_LENGTH];


__constant__ int d_c0[STATE_NUM][2];
__constant__ int d_c1[STATE_NUM][2];

__constant__ int d_lookup1[STATE_NUM];
__constant__ int d_lookup2[STATE_NUM];
