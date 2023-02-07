__global__ void sum2(char *charArr, int n, int stride, char startingValue, char *base) {
    long elems_per_thread = (n+stride-1) / stride;
    char sum = startingValue;
    for (long i = 0; i < stride-1; i++) {
        base[i] = sum;
        sum += charArr[elems_per_thread * (i+1) - 1];
    }
    base[stride-1] = sum;
}