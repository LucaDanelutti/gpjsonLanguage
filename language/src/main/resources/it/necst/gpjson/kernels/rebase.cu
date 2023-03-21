__global__ void f(int *data, int n, int *base) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    int abase = (blockIdx.x == 0) ? 0 : base[blockIdx.x-1];

    if (i*2 < n)
        data[i*2] = baseValue + data[i*2] + abase;
    if (i*2+1 < n)
        data[i*2+1] = baseValue + data[i*2+1] + abase;
}