__global__ void f(char *charArr, int n, int stride, char startingValue, char *base) {
    long elemsPerThread = (n+stride-1) / stride;
    char sum = startingValue;
    for (long i = 0; i < stride-1; i++) {
        base[i] = sum;
        sum += charArr[elemsPerThread * (i+1) - 1];
    }
    base[stride-1] = sum;
}