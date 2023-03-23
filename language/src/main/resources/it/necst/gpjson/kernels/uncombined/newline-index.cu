__global__ void f(char *file, int fileSize, int *newlineCountIndex, long *newlineIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int offset = newlineCountIndex[index];

  long charsPerThread = (fileSize+stride-1) / stride;
  long start = index * charsPerThread;
  long end = start + charsPerThread;

  for (int i = start; i < end && i < fileSize; i += 1) {
    if (file[i] == '\n') {
      newlineIndex[offset++] = i;
    }
  }
}
