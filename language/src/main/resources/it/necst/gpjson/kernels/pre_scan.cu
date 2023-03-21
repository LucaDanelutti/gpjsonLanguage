__global__ void f(int *data, int n, int rep) {
    extern __shared__ int sdata[];
    __shared__ int sum;

    for (int r=0; r < rep; r++) {
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*blockDim.x + blockDim*r + threadIdx.x;
        sdata[tid*2] = (i*2 < n) ? data[i*2] : 0;
        sdata[tid*2+1] = (i*2+1 < n) ? data[i*2+1] : 0;
        sdata[tid*2] += sum;
        sdata[tid*2+1] += sum;
        __syncthreads();

        int stride = 1;
        while(stride <= blockDim.x) {
            int index = (threadIdx.x+1)*stride*2 - 1;
            if (index < 2*blockDim.x)
                sdata[index] += sdata[index-stride];
            stride = stride*2;
        __syncthreads();
        }

        if (i*2 < n)
            data[i*2] = sdata[tid*2];
        if (i*2+1 < n)
            data[i*2+1] = sdata[tid*2+1];
        __syncthreads();

        if (threadIdx.x == blockDim.x-1)
            sum = sdata[tid*2+1];
        __syncthreads();
    }
}