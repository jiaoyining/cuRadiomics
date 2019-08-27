#ifndef MYUTILS_H
#define MYUTILS_H

#define SAFE_SIZE 1024 * 1

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <set>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

void HandleError(cudaError_t err, const char *file, int line);

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define START_TIMER(name)         \
	struct timespec name##_start; \
	clock_gettime(CLOCK_MONOTONIC, &name##_start);

#define STOP_TIMER(name)        \
	struct timespec name##_end; \
	clock_gettime(CLOCK_MONOTONIC, &name##_end);

#define GET_TIME(name) \
	name##_end.tv_sec - name##_start.tv_sec + (name##_end.tv_nsec - name##_start.tv_nsec) / 1000000000.

#define PRINT_TIME(name) \
	printf("%s\t%0.6f\r\n", #name, name##_end.tv_sec - name##_start.tv_sec + (name##_end.tv_nsec - name##_start.tv_nsec) / 1000000000.);

	using namespace std;

typedef unsigned long long TIME_USEC;

void print2DData(FILE *output, double *data, long width, long height);
void print2DData(FILE *output, float *data, long width, long height);

TIME_USEC get_current_usec();
double measure_nsec(struct timespec start);
typedef struct timespec TIME_NSEC;

char *getArgumentValue(int argc, char **argv, char *argName);

int msleep(unsigned long milisec);

long safeSize(long size);


// These names may vary by implementation
#define LINEAR_CONGRUENTIAL_ENGINE linear_congruential_engine
//#define LINEAR_CONGRUENTIAL_ENGINE linear_congruential_engine
#define UNIFORM_INT_DISTRIBUTION uniform_int_distribution
//#define UNIFORM_INT_DISTRIBUTION uniform_int

typedef unsigned int uint;
typedef unsigned long ulong;

typedef set<int> node;

#endif
