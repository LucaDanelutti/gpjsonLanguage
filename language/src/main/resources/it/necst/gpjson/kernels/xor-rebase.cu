__global__ void f(char *charArr, int n, char *base) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long elemsPerThread = (n+stride-1) / stride;

    long start = index * elemsPerThread;
    long end = start + elemsPerThread;

    for (long i = start; i < end && i < n; i++) {
        charArr[i] ^= base[index];
    }
}