#include "utils.h"



void load_data(int *image, int *mask, int *size)
{
    FILE *filePtr1, *filePtr2;
    filePtr1 = fopen("/home/jyn/cuda-workspace/r/Brats18_2013_2_1/Brats18_2013_2_1_flair.nii.gz", "r");
    filePtr2 = fopen("/home/jyn/cuda-workspace/r/Brats18_2013_2_1/Brats18_2013_2_1_seg.nii.gz", "r");

    int zz = 0;

    for (zz = 0; zz < size[0] * size[1] * size[2]; zz++)
    {

        int m;
        int n;
        fscanf(filePtr1, "%d ", &m);
        fscanf(filePtr2, "%d ", &n);

        image[zz] = m;
        mask[zz] = n;
    }

    fclose(filePtr1);
    fclose(filePtr2);
}


void print2DData(FILE *output, double *data, long width, long height)
{
	long k = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			fprintf(output, " %2.1f\t", data[k++]);
		}
		fprintf(output, "\n");
	}
	fprintf(output, "\n");
}

void print2DData(FILE *output, float *data, long width, long height)
{
	long k = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			fprintf(output, " %2.1f\t", data[k++]);
		}
		fprintf(output, "\n");
	}
	fprintf(output, "\n");
}

TIME_USEC get_current_usec()
{
	struct timeval time;
	gettimeofday(&time, NULL);

	TIME_USEC msec;
	msec = (time.tv_sec * 1000000);
	msec += (time.tv_usec);
	return msec;
}
//
//double measure_nsec(struct timespec start){
//	struct timespec end;
//	clock_gettime(CLOCK_MONOTONIC, &end);
//	return end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.;
//}

char *getArgumentValue(int argc, char **argv, char *argName)
{
	char *result = NULL;

	for (int i = 0; i < argc; i++)
	{
		char *substr = strstr(argv[i], argName);
		if (substr != NULL)
		{
			result = strtok(substr, "=");
			if (result != NULL)
			{
				result = strtok(NULL, "=");
			}
			break;
		}
	}

	return result;
}

int msleep(unsigned long milisec)
{
	struct timespec req = {0};
	time_t sec = (int)(milisec / 1000);
	milisec = milisec - (sec * 1000);
	req.tv_sec = sec;
	req.tv_nsec = milisec * 1000000L;
	while (nanosleep(&req, &req) == -1)
		continue;
	return 1;
}

long safeSize(long size)
{
	if (size < SAFE_SIZE)
		size = SAFE_SIZE;

	return size;
}

void setIntValue(int *lvalue, char *rvalue, int defaultValue)
{
	if (rvalue != NULL)
	{
		*lvalue = atoi(rvalue);
	}
	//	else{
	//		*lvalue = defaultValue;
	//	}
}

void setFloatValue(float *lvalue, char *rvalue, float defaultValue)
{
	if (rvalue != NULL)
	{
		*lvalue = atof(rvalue);
	}
	//	else{
	//		*lvalue = defaultValue;
	//	}
}



void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			   file, line);
		exit(EXIT_FAILURE);
	}
}
