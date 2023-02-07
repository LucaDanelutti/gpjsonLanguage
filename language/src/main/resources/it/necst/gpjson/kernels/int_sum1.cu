__global__ void sum1(int *intArr, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long elems_per_thread = (n+stride-1) / stride;

    long start = index * elems_per_thread;
    long end = start + elems_per_thread;

    int sum = 0;
    for (long i = start; i < end && i < n; i++) {
        sum += intArr[i];
        intArr[i] = sum;
    }
}