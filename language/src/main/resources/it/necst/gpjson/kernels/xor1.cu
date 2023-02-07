__global__ void xor1(char *charArr, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long elems_per_thread = (n+stride-1) / stride;

    long start = index * elems_per_thread;
    long end = start + elems_per_thread;

    char prev = 0;
    for (long i = start; i < end && i < n; i++) {
        prev ^= charArr[i];
        charArr[i] = prev;
    }
}