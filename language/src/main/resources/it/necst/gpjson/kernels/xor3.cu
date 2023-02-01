__global__ void xor3(char *arr, int n, char *base) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long elems_per_thread = (n+stride-1) / stride;

    long start = index * elems_per_thread;
    long end = start + elems_per_thread;

    for (long i = start; i < end && i < n; i++) {
        arr[i] ^= base[index];
    }
}