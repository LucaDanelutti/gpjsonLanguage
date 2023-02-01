__global__ void sum2(int *arr, int n, int stride, int *base) {
    long elems_per_thread = (n+stride-1) / stride;
    int sum = 1;
    for (long i = 0; i < stride-1; i++) {
        base[i] = sum;
        sum += arr[elems_per_thread * (i+1) - 1];
    }
    base[stride-1] = sum;
}