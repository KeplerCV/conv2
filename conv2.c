#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <minmax.h>
//#include "vld.h"

/* the alignment of all the allocated buffers */
#define  KEPLORE_MALLOC_ALIGN    64


#define KEPLORE_S08 1
#define KEPLORE_U08 2
#define KEPLORE_S16 3
#define KEPLORE_U16 4
#define KEPLORE_S32 5
#define KEPLORE_U32 6
#define KEPLORE_F32 7
#define KEPLORE_F64 8

//http://blog.csdn.net/tcx19900712/article/details/18218225
//align memory block reference linking
void* aligned_malloc(size_t size, int alignment)
{   
	const int pointerSize = sizeof(void*);
	const int requestedSize = size + alignment - 1 + pointerSize; 
	void* raw = malloc(requestedSize);   
	uintptr_t start = (uintptr_t)raw + pointerSize;
	void* aligned = (void*)((start + alignment - 1) & ~(alignment - 1));
	*(void**)((uintptr_t)aligned - pointerSize) = raw;
	return aligned;
}

void aligned_free(void* aligned)
{ 
	void* raw = *(void**)((uintptr_t)aligned - sizeof(void*));
	free(raw);
}

int isAligned(void* data, int alignment)
{ 
	return ((uintptr_t)data & (alignment - 1)) == 0;
}


typedef struct _Mat
{
	int type;
	int step;
	int channels;
	union
	{
		unsigned char* ptr;
		short* s;
		int* i;
		float* fl;
		double* db;
	} data;
	int rows;
	int cols;
}
Mat;

typedef enum {
	CONV2_SHAPE_FULL = 0,
	CONV2_SHAPE_SAME = 1,
	CONV2_SHAPE_VALID = 2
}CONV_SHAPE;

//http://blog.csdn.net/celerychen2009/article/details/38852105
//Matlab conv2 implement reference linking
int conv2(Mat *src, Mat *dst, Mat *kernel, CONV_SHAPE shape)
{
	int src_rows = src->rows;
	int src_cols = src->cols;
	int kernel_rows = kernel->rows;
	int kernel_cols = kernel->cols;
	int dst_rows = 0;
	int dst_cols = 0;
	int edge_rows = 0;
	int edge_cols = 0;
	int i, j, kernel_i, kernel_j, src_i, src_j;
	double *p_src = NULL;
	double *p_dst = NULL;
	double *p_kernel = NULL;
	double *p_dst_line_i = NULL;
	double *ptr_src_line_i = NULL;
    double *ptr_kernel_line_i = NULL;
	double sum = 0;

	p_src = src->data.db;
	p_dst = dst->data.db;
	p_kernel = kernel->data.db;

	switch (shape) 
	{
	case CONV2_SHAPE_FULL:

		dst_rows = src_rows + kernel_rows - 1;
		dst_cols = src_cols + kernel_cols - 1;
		edge_rows = kernel_rows - 1;
		edge_cols = kernel_cols - 1;
		break;

	case CONV2_SHAPE_SAME:

		dst_rows = src_rows;
		dst_cols = src_cols;
		edge_rows = (kernel_rows - 1) / 2;
		edge_cols = (kernel_cols - 1) / 2;
		break;

	case CONV2_SHAPE_VALID:

		dst_rows = src_rows - kernel_rows + 1;
		dst_cols = src_cols - kernel_cols + 1;
		edge_rows = edge_cols = 0;
		break;
	}

	for (i = 0; i < dst_rows; i++)
	{
		p_dst_line_i = (double *)(p_dst + dst_cols * i);
		for (j = 0; j < dst_cols; j++)
		{
			sum = 0;

			kernel_i = kernel_rows - 1 - max(0, edge_rows - i);
			src_i = max(0, i - edge_rows);
			for (; (kernel_i >= 0) && (src_i < src_rows); kernel_i--, src_i++)
			{
				kernel_j = kernel_cols - 1 - max(0, edge_cols - j);
				src_j = max(0, j - edge_cols);

				ptr_src_line_i = (double*)(p_src + src_cols * src_i);
				ptr_kernel_line_i = (double*)(p_kernel + kernel_cols * kernel_i);

				ptr_src_line_i += src_j;
				ptr_kernel_line_i += kernel_j;

				for (; kernel_j >= 0 && src_j < src_cols; kernel_j--, src_j++) {
					sum += *ptr_src_line_i++ * *ptr_kernel_line_i--;
				}
			}
			p_dst_line_i[j] = sum;
		}
	}

	return 0;
}

int main(int argc, char *argv[])
{
	int i, j;
	int dst_size = 0;
	Mat src, dst, kernel;
	double kernel_data[3][3] = { { 1, 3, 1 },
	                             { 0, 5, 0 },
	                             { 2, 1, 2 } };
	double src_data[5][5] = { { 17, 24, 1, 8, 15 },
							  { 23, 5, 7, 14, 16 },
							  { 4, 6, 13, 20, 22 },
							  { 10, 12, 19, 21, 3 },
							  { 11, 18, 25, 2, 9 } };
	double *dst_data = NULL;
	int conv_shape = CONV2_SHAPE_VALID; //output type
	if (conv_shape == CONV2_SHAPE_FULL)
	{
		dst_size = 7;
	}
	else if (conv_shape == CONV2_SHAPE_SAME)
	{
		dst_size = 5;
	}
	else if(conv_shape == CONV2_SHAPE_VALID)
	{
		dst_size = 3;
	}
	else
	{
		return -1;
	}

	dst_data = aligned_malloc(dst_size * dst_size *sizeof(double), KEPLORE_MALLOC_ALIGN);

	src.data.db = src_data[0];
	dst.data.db = dst_data;
	kernel.data.db = kernel_data[0];

	src.cols = 5;
	src.rows = 5;
	src.type = KEPLORE_F64;
	src.step = 5;
	src.channels = 1;

	kernel.cols = 3;
	kernel.rows = 3;
	kernel.type = KEPLORE_F64;
	kernel.step = 3;
	kernel.channels = 1;

	if (conv_shape == CONV2_SHAPE_FULL)
	{
		dst.cols = 7;
		dst.rows = 7;
		dst.type = KEPLORE_F64;
		dst.step = 7;
		dst.channels = 1;
	}
	else if (conv_shape == CONV2_SHAPE_SAME)
	{
		dst.cols = 5;
		dst.rows = 5;
		dst.type = KEPLORE_F64;
		dst.step = 5;
		dst.channels = 1;
	}
	else if (conv_shape == CONV2_SHAPE_VALID)
	{
		dst.cols = 3;
		dst.rows = 3;
		dst.type = KEPLORE_F64;
		dst.step = 3;
		dst.channels = 1;
	}
	else
	{
		return -1;
	}

	conv2(&src, &dst, &kernel, conv_shape);

	for (i = 0; i < dst.rows; i++)
	{
		for (j = 0; j < dst.cols; j++)
		{
			printf("%lf ", dst.data.db[i * dst.cols + j]);
		}
		printf("\n");
	}

	aligned_free(dst_data);
	dst_data = NULL;
	return 0;
}
