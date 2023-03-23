__global__ void f(char *file, int fileSize, long *escapeIndex, long *quoteIndex, char *quoteCarryIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize+stride-1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  long escaped = 0;
  long quote = 0;
  int quoteCount = 0;

  int final_loop_iteration = end;
  if (fileSize < end) {
    final_loop_iteration = fileSize;
  }

  for (long i = start; i < end && i < fileSize; i += 1) {
    long offsetInBlock = i % 64;

    if (offsetInBlock == 0) {
      escaped = escapeIndex[i / 64];
    }

    if (file[i] == '"') {
      if ((escaped & (1L << offsetInBlock)) == 0) {
        quote = quote | (1L << offsetInBlock);
        quoteCount++;
      }
    }

    if (offsetInBlock == 63L) {
      quoteIndex[i / 64] = quote;
      quote = 0;
    }
  }

  if (fileSize <= end && (fileSize - 1) % 64 != 63L && fileSize - start > 0) {
    quoteIndex[(fileSize - 1) / 64] = quote;
  }

  quoteCarryIndex[index] = quoteCount & 1;
}
