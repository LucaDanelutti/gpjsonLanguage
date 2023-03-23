__global__ void f(char *file, int fileSize, int *newlineCountIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  long charsPerThread = (fileSize+stride-1) / stride;
  long start = index * charsPerThread;
  long end = start + charsPerThread;

  int count = 0;
  for (long i = start; i < end && i < fileSize; i += 1) {
    if (file[i] == '\n') {
      count += 1;
    }
  }

  newlineCountIndex[index] = count;
}
