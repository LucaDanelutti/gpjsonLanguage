__global__ void f(char *charArr, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long elemsPerThread = (n+stride-1) / stride;

    long start = index * elemsPerThread;
    long end = start + elemsPerThread;

    char prev = 0;
    for (long i = start; i < end && i < n; i++) {
        prev ^= charArr[i];
        charArr[i] = prev;
    }
}