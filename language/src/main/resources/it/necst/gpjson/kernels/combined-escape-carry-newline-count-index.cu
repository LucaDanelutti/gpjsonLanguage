__global__ void f(char *file, int fileSize, char *escapeCarry, int *newlineCount) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize+stride-1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  char carry = 0;
  int count = 0;

  for (int i = start; i < end && i < fileSize; i += 1) {
    char value = file[i];
    if (value == '\\') {
      carry = 1 ^ carry;
    } else {
      carry = 0;
    }

    if (value == '\n') {
     count += 1;
    }
  }

  newlineCount[index] = count;
  escapeCarry[index] = carry;
}
