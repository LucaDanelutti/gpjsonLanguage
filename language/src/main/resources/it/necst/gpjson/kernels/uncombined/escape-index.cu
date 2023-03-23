__global__ void f(char *file, long fileSize, bool *escapeCarryIndex, long *escapeIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize+stride-1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  bool carry = index == 0 ? false : escapeCarryIndex[index - 1];

  long escape = 0;

  int escapeCount = 0;
  int totalCount = end - start;

  for (long i = start; i < end && i < fileSize; i += 1) {
    if (carry == 1) {
      escape = escape | (1L << (i % 64));
    }

    if (file[i] == '\\') {
      escapeCount++;
      carry = carry ^ 1;
    } else {
      carry = 0;
    }

    if (i % 64 == 63) {
      escapeIndex[i / 64] = escape;
      escape = 0;
    }
  }

  if (fileSize <= end && (fileSize - 1) % 64 != 63L && fileSize - start > 0) {
    escapeIndex[(fileSize - 1) / 64] = escape;
  }

  assert(escapeCount != totalCount);
}
