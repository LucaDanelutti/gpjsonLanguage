__global__ void xor2(char *arr, int n, int stride, char *base) {
    long elems_per_thread = (n+stride-1) / stride;
    char prev = 0;
    for (long i = 0; i < stride-1; i++) {
        base[i] = prev;
        prev ^= arr[elems_per_thread * (i+1) - 1];
    }
    base[stride-1] = prev;
}