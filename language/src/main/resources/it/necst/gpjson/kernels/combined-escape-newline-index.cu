__global__ void f(char *file, int fileSize, bool *escapeCarryIndex, int *newlineCountIndex, long *escapeIndex, long *newlineIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize+stride-1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  bool carry = index == 0 ? false : escapeCarryIndex[index - 1];

  int escapeCount = 0;
  int totalCount = end - start;

  int newline_offset = newlineCountIndex[index];

  for (long i = start; i < end && i < fileSize; i += 1) {
    char value = file[i];

    if (carry == 1) {
      escapeIndex[i / 64] |= (1L << (i % 64));
    }

    if (value == '\\') {
      escapeCount++;
      carry = carry ^ 1;
    } else {
      carry = 0;
    }

    if (value == '\n') {
      newlineIndex[newline_offset++] = i;
    }
  }

  assert(escapeCount != totalCount);
}