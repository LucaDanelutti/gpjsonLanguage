__global__ void sum2(int *intArr, int n, int stride, int startingValue, int *base) {
    long elems_per_thread = (n+stride-1) / stride;
    int sum = startingValue;
    for (long i = 0; i < stride-1; i++) {
        base[i] = sum;
        sum += intArr[elems_per_thread * (i+1) - 1];
    }
    base[stride-1] = sum;
}