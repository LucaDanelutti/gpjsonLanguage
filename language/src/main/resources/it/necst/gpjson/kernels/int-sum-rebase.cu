__global__ void f(int *intArr, int n, int *base, int offset, int *intNewArr) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long elemsPerThread = (n+stride-1) / stride;

    long start = index * elemsPerThread;
    long end = start + elemsPerThread;

    for (long i = start; i < end && i < n; i++) {
        intNewArr[i+offset] = intArr[i] + base[index];
    }
}