__global__ void sum3(int *intArr, int n, int *base, int offset, int *intNewArr) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long elems_per_thread = (n+stride-1) / stride;

    long start = index * elems_per_thread;
    long end = start + elems_per_thread;

    for (long i = start; i < end && i < n; i++) {
        intNewArr[i+offset] = intArr[i] + base[index];
    }
}