__global__ void f(char *file, int fileSize, long *stringIndex, char *leveledBitmapsAuxIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize+stride-1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  long string = 0;

  signed char level = 0;

  for (long i = start; i < end && i < fileSize; i += 1) {
    long offsetInBlock = i % 64;

    if (offsetInBlock == 0) {
      string = stringIndex[i / 64];
    }

    if ((string & (1L << offsetInBlock)) != 0) {
      continue;
    }

    char value = file[i];

    if (value == '{' || value == '[') {
      level++;
    } else if (value == '}' || value == ']') {
      level--;
    }
  }

  leveledBitmapsAuxIndex[index] = level;
}
