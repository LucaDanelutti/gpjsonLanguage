__global__ void f(int *data, int n, int *sum) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid*2] = (i*2 < n) ? data[i*2] : 0;
    sdata[tid*2+1] = (i*2+1 < n) ? data[i*2+1] : 0;
    __syncthreads();

    int stride = blockDim.x;
    while(stride > 0) {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if (index+stride < 2*blockDim.x) {
            sdata[index+stride] += sdata[index];
        }
        stride = stride / 2;
        __syncthreads();
    }

    if (threadIdx.x == 0)
        sum[blockIdx.x] = sdata[2*blockDim.x-1];
    if (i*2 < n)
        data[i*2] = sdata[tid*2];
    if (i*2+1 < n)
        data[i*2+1] = sdata[tid*2+1];
}