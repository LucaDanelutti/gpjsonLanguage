__global__ void f(int *intArr, int n, int stride, int startingValue, int *base) {
    long elemsPerThread = (n+stride-1) / stride;
    int sum = startingValue;
    for (long i = 0; i < stride-1; i++) {
        base[i] = sum;
        sum += intArr[elemsPerThread * (i+1) - 1];
    }
    base[stride-1] = sum;
}