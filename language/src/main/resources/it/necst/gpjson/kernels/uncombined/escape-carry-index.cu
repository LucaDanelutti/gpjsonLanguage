__global__ void f(char *file, int fileSize, char *escapeCarryIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize+stride-1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  char carry = 0;

  for (int i = start; i < end && i < fileSize; i += 1) {
    if (file[i] == '\\') {
      carry = 1 ^ carry;
    } else {
      carry = 0;
    }
  }

  escapeCarryIndex[index] = carry;
}
