__global__ void spmv_ell_kernel(const int num_rows, const int num_cols, const int num_cols_per_row, const int * indices, const float * data, const float * x, const float * y)
{
	int row = blockDim * blockIdx.x + threadIdx.x;
	if(row < num_rows)
	{
		float dot = 0;
		for (int i = 0; i < num_cols_per_row; i++)
		{
			int col = indices[num_rows * i + row];
			float val = data[num_rows * i + row];
			dot += val * x[col];
		}
		y[row] += dot;
	}	
}

