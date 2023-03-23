__global__ void f(long *stringIndex, int stringIndexSize, char *quoteCarryIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int elemsPerThread = (stringIndexSize + stride - 1) / stride;
  int start = index * elemsPerThread;
  int end = start + elemsPerThread;

  long bitString = index > 0 && quoteCarryIndex[index - 1] == 1 ? 0xffffffffffffffffL : 0;

  for (int i = start; i < end && i < stringIndexSize; i += 1) {
    long quotes = stringIndex[i];

    // https://github.com/simdjson/simdjson/blob/cfc965ff9ada688cf5950da829331b28dfcb949f/include/simdjson/arm64/bitmask.h
    quotes ^= quotes << 1;
    quotes ^= quotes << 2;
    quotes ^= quotes << 4;
    quotes ^= quotes << 8;
    quotes ^= quotes << 16;
    quotes ^= quotes << 32;

    quotes = quotes ^ bitString;

    stringIndex[i] = quotes;

    bitString = quotes >> 63;
  }
}