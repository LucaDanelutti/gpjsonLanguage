__global__ void sum3(int *arr, int n, int *base, int offset, int *newArr) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long elems_per_thread = (n+stride-1) / stride;

    long start = index * elems_per_thread;
    long end = start + elems_per_thread;

    for (long i = start; i < end && i < n; i++) {
        newArr[i+offset] = arr[i] + base[index];
    }
}